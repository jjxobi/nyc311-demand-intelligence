-- Staging model: cleans and types raw 311 data.
-- One row per unique service request (deduplicated on unique_key).

with source as (

    select * from {{ source('raw', 'raw_311_requests') }}

),

deduplicated as (

    -- Raw ingestion can produce duplicate unique_keys if a date range was
    -- re-fetched (e.g. overlapping incremental loads). Keep the latest
    -- created_date per unique_key as the source of truth.
    select *,
        row_number() over (
            partition by unique_key
            order by created_date desc
        ) as _row_num
    from source

),

cleaned as (

    select
        unique_key,
        created_date,

        -- closed_date arrives as VARCHAR from the raw layer because some
        -- requests are still open (null). Explicit cast to TIMESTAMP here
        -- rather than relying on inference, which produced a string column.
        try_cast(closed_date as timestamp) as closed_date,

        -- Standardize complaint_type casing/whitespace. Raw 311 data has
        -- inconsistent labels (e.g. trailing spaces, inconsistent casing
        -- across years) -- documented data quality issue.
        trim(complaint_type) as complaint_type,

        -- Borough sometimes arrives as 'Unspecified' or null depending on
        -- source system. Normalize both to a single NULL-like flag value
        -- so downstream aggregations don't silently split this group.
        case
            when borough is null or upper(trim(borough)) = 'UNSPECIFIED'
                then 'UNSPECIFIED'
            else upper(trim(borough))
        end as borough,

        trim(agency) as agency,
        trim(status) as status,

        -- Zip codes occasionally arrive with non-numeric junk or wrong
        -- length. Keep only plausible 5-digit NYC zips, null out the rest
        -- rather than silently keeping bad data.
        case
            when incident_zip is not null
                and length(trim(incident_zip)) = 5
                and try_cast(trim(incident_zip) as integer) is not null
                then trim(incident_zip)
            else null
        end as incident_zip

    from deduplicated
    where _row_num = 1

)

select * from cleaned