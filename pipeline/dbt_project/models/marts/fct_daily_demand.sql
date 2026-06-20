/*
Fact table: daily demand by category and borough, joined to category
dimension. This is where the primary table forecasting, the dashboard, and
the LLM assistant all read from. Grain: one row per
(date, complaint_type, borough), matching the intermediate layer.
*/

with daily_volume as (

    select * from {{ ref('int_daily_volume_by_category') }}

),

categories as (

    select * from {{ ref('dim_complaint_category') }}

),

final as (

    select
        daily_volume.request_date,
        categories.complaint_category_id,
        daily_volume.complaint_type,
        categories.category_group,
        daily_volume.borough,
        daily_volume.request_count,
        daily_volume.closed_count,
        daily_volume.avg_resolution_hours,
        daily_volume.same_day_batch_closed_count,

        -- Calendar features computed once here so every downstream
        -- consumer (forecasting, dashboard) gets them for free instead
        -- of recomputing date logic independently in multiple places.
        extract(dow from daily_volume.request_date) as day_of_week,
        case
            when extract(dow from daily_volume.request_date) in (0, 6)
                then true
            else false
        end as is_weekend,
        extract(month from daily_volume.request_date) as month,
        extract(year from daily_volume.request_date) as year

    from daily_volume
    left join categories
        on daily_volume.complaint_type = categories.complaint_type

)

select * from final