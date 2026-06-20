/* 
Dimension table: one row per distinct complaint category.
Provides a stable surface for category-level metadata and labeling,
decoupled from the fact table so labels can be enriched later
(e.g. grouping categories into higher-level themes) without touching
the fact table at all.
*/

with categories as (

    select distinct complaint_type
    from {{ ref('stg_311_requests') }}

),

enriched as (

    select
        {{ dbt_utils.generate_surrogate_key(['complaint_type']) }} as complaint_category_id,
        complaint_type,

        -- Coarse grouping, useful for dashboard filtering/rollups.
        -- Mapping kept intentionally simple and documented inline.
        case
            when complaint_type ilike '%noise%' then 'Noise'
            when complaint_type in ('Illegal Parking', 'Blocked Driveway') then 'Parking & Traffic'
            when complaint_type = 'Traffic Signal Condition' then 'Parking & Traffic'
            when complaint_type in ('HEAT/HOT WATER', 'PLUMBING', 'PAINT/PLASTER') then 'Housing'
            when complaint_type in ('Water System', 'Sewer') then 'Infrastructure - Water'
            when complaint_type in ('Street Condition', 'Damaged Tree') then 'Infrastructure - Street'
            when complaint_type = 'Sanitation Condition' then 'Sanitation'
            when complaint_type = 'Air Quality' then 'Environment'
            else 'Other'
        end as category_group

    from categories

)

select * from enriched