/*
Intermediate model: daily request volume by complaint category and borough.
Grain: one row per (date, complaint_type, borough).
This will be the core aggregation that forecasting models train on.
*/

with staged as (

    select * from {{ ref('stg_311_requests') }}

),

daily_counts as (

    select
        date_trunc('day', created_date) as request_date,
        complaint_type,
        borough,

        count(*) as request_count,

        -- Can be useful downstream for "how long do these typically take to close"
        count(*) filter (where closed_date is not null) as closed_count,

        -- Exclude same-timestamp closures (closed_date = created_date), which
        -- are bulk/batch-closed records (concentrated in DOT and DEP agencies,
        -- ~70K rows in the full dataset) rather than genuinely instant
        -- resolutions. Including them would artificially deflate this average.
        avg(
            case
                when closed_date is not null and closed_date > created_date
                    then date_diff('hour', created_date, closed_date)
            end
        ) as avg_resolution_hours,

        -- Tracked separately so this data characteristic stays visible
        -- downstream rather than silently disappearing.
        count(*) filter (where closed_date is not null and closed_date <= created_date) as same_day_batch_closed_count

    from staged
    group by 1, 2, 3

)

select * from daily_counts
order by request_date, complaint_type, borough