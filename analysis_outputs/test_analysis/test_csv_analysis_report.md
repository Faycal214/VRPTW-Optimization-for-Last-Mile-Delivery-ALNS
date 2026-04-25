# VRPTW test CSV analysis

## Key insights
- Best average objective: Mixed_tight with 4563.67.
- Highest average objective: Random_large with 15926.90.
- Highest feasibility rate: Clustered_large at 100.00%.
- For Clustered, tight instances are not worse on objective on average: objective changes by -3288.42.
- For Clustered, route count changes by 12.33 from large to tight.
- For Clustered, total time changes by -3689.13 from large to tight.
- For Random, tight instances are not worse on objective on average: objective changes by -9882.60.
- For Random, route count changes by 14.33 from large to tight.
- For Random, total time changes by -1614.88 from large to tight.
- For Mixed, tight instances are not worse on objective on average: objective changes by -6801.37.
- For Mixed, route count changes by 9.33 from large to tight.
- For Mixed, total time changes by -5438.70 from large to tight.
- Overall test feasible rate: 100.00%, mean objective: 10028.17.

## Overall test overview
|   n_instances |   feasible_rate |   mean_routes |   mean_customers |   mean_distance |   mean_time |   mean_objective |   median_objective |
|--------------:|----------------:|--------------:|-----------------:|----------------:|------------:|-----------------:|-------------------:|
|            18 |               1 |       14.8889 |              200 |         5409.87 |     15633.9 |          10028.2 |            9472.92 |

## Family summary
| family          |   n_instances |   feasible_rate |   mean_total_routes |   mean_num_customers |   mean_load_variance |   mean_spatial_variance |   mean_total_distance |   mean_total_time |   mean_vehicles_used |   mean_objective |   median_objective |   min_objective |   max_objective |
|:----------------|--------------:|----------------:|--------------------:|---------------------:|---------------------:|------------------------:|----------------------:|------------------:|---------------------:|-----------------:|-------------------:|----------------:|----------------:|
| Clustered_large |             3 |               1 |             8.33333 |                  200 |            18760.9   |                40.199   |               4619.61 |          26293.4  |              8.33333 |         12778.8  |           14331    |         9520.65 |        14484.7  |
| Clustered_tight |             3 |               1 |            20.6667  |                  200 |             1155.4   |                 7.26576 |               4105.24 |          22604.2  |             20.6667  |          9490.36 |            9425.19 |         9136.44 |         9909.45 |
| Random_large    |             3 |               1 |             8.33333 |                  200 |            82268.9   |               172.52    |               6330.61 |          12261.5  |              8.33333 |         15926.9  |           15374.5  |         6117.98 |        26288.3  |
| Random_tight    |             3 |               1 |            22.6667  |                  200 |             2812.29  |                11.7784  |               6104.96 |          10646.6  |             22.6667  |          6044.3  |            6184.73 |         5119.63 |         6828.53 |
| Mixed_large     |             3 |               1 |            10       |                  200 |            39593.5   |                87.8585  |               6421.52 |          13718.3  |             10       |         11365    |           10793.7  |        10221.2  |        13080.3  |
| Mixed_tight     |             3 |               1 |            19.3333  |                  200 |              494.358 |                 6.21659 |               4877.26 |           8279.62 |             19.3333  |          4563.67 |            4565.92 |         4448.36 |         4676.73 |

## Large vs tight comparison inside each distribution
| distribution   | large_family    | tight_family    |   large_instances |   tight_instances |   large_total_routes |   tight_total_routes |   diff_total_routes |   rel_diff_total_routes |   large_num_customers |   tight_num_customers |   diff_num_customers |   rel_diff_num_customers |   large_load_variance |   tight_load_variance |   diff_load_variance |   rel_diff_load_variance |   large_spatial_variance |   tight_spatial_variance |   diff_spatial_variance |   rel_diff_spatial_variance |   large_total_distance |   tight_total_distance |   diff_total_distance |   rel_diff_total_distance |   large_total_time |   tight_total_time |   diff_total_time |   rel_diff_total_time |   large_vehicles_used |   tight_vehicles_used |   diff_vehicles_used |   rel_diff_vehicles_used |   large_objective |   tight_objective |   diff_objective |   rel_diff_objective |
|:---------------|:----------------|:----------------|------------------:|------------------:|---------------------:|---------------------:|--------------------:|------------------------:|----------------------:|----------------------:|---------------------:|-------------------------:|----------------------:|----------------------:|---------------------:|-------------------------:|-------------------------:|-------------------------:|------------------------:|----------------------------:|-----------------------:|-----------------------:|----------------------:|--------------------------:|-------------------:|-------------------:|------------------:|----------------------:|----------------------:|----------------------:|---------------------:|-------------------------:|------------------:|------------------:|-----------------:|---------------------:|
| Clustered      | Clustered_large | Clustered_tight |                 3 |                 3 |              8.33333 |              20.6667 |            12.3333  |                1.48     |                   200 |                   200 |                    0 |                        0 |               18760.9 |              1155.4   |             -17605.5 |                -0.938415 |                  40.199  |                  7.26576 |                -32.9333 |                   -0.819255 |                4619.61 |                4105.24 |              -514.375 |                 -0.111346 |            26293.4 |           22604.2  |          -3689.13 |             -0.140307 |               8.33333 |               20.6667 |             12.3333  |                 1.48     |           12778.8 |           9490.36 |         -3288.42 |            -0.257335 |
| Random         | Random_large    | Random_tight    |                 3 |                 3 |              8.33333 |              22.6667 |            14.3333  |                1.72     |                   200 |                   200 |                    0 |                        0 |               82268.9 |              2812.29  |             -79456.6 |                -0.965816 |                 172.52   |                 11.7784  |               -160.742  |                   -0.931728 |                6330.61 |                6104.96 |              -225.654 |                 -0.035645 |            12261.5 |           10646.6  |          -1614.88 |             -0.131704 |               8.33333 |               22.6667 |             14.3333  |                 1.72     |           15926.9 |           6044.3  |         -9882.6  |            -0.620498 |
| Mixed          | Mixed_large     | Mixed_tight     |                 3 |                 3 |             10       |              19.3333 |             9.33333 |                0.933333 |                   200 |                   200 |                    0 |                        0 |               39593.5 |               494.358 |             -39099.1 |                -0.987514 |                  87.8585 |                  6.21659 |                -81.6419 |                   -0.929243 |                6421.52 |                4877.26 |             -1544.27  |                 -0.240483 |            13718.3 |            8279.62 |          -5438.7  |             -0.396455 |              10       |               19.3333 |              9.33333 |                 0.933333 |           11365   |           4563.67 |         -6801.37 |            -0.598447 |

## Best instances
| instance            | feasible   |   total_routes |   num_customers | errors   | missing_customers   | duplicate_customers   |   load_variance |   spatial_variance |   total_distance |   total_time |   vehicles_used |   objective | family       |
|:--------------------|:-----------|---------------:|----------------:|:---------|:--------------------|:----------------------|----------------:|-------------------:|-----------------:|-------------:|----------------:|------------:|:-------------|
| Mixed_tight_200_13  | True       |             19 |             200 | []       | []                  | []                    |         471.562 |            6.87535 |          4529.97 |      8224.93 |              19 |     4448.36 | Mixed_tight  |
| Mixed_tight_200_15  | True       |             20 |             200 | []       | []                  | []                    |         686.79  |            5.165   |          4893.19 |      8170.83 |              20 |     4565.92 | Mixed_tight  |
| Mixed_tight_200_14  | True       |             19 |             200 | []       | []                  | []                    |         324.72  |            6.60942 |          5208.61 |      8443.09 |              19 |     4676.73 | Mixed_tight  |
| Random_tight_200_8  | True       |             19 |             200 | []       | []                  | []                    |         612.936 |            9.8892  |          5373.72 |      9491.02 |              19 |     5119.63 | Random_tight |
| Random_large_200_10 | True       |              6 |             200 | []       | []                  | []                    |         644.917 |            9.94444 |          6362.93 |     12368.8  |               6 |     6117.98 | Random_large |

## Worst instances
| instance              | feasible   |   total_routes |   num_customers | errors   | missing_customers   | duplicate_customers   |   load_variance |   spatial_variance |   total_distance |   total_time |   vehicles_used |   objective | family          |
|:----------------------|:-----------|---------------:|----------------:|:---------|:--------------------|:----------------------|----------------:|-------------------:|-----------------:|-------------:|----------------:|------------:|:----------------|
| Random_large_200_12   | True       |              6 |             200 | []       | []                  | []                    |        176057   |           358.611  |          6132.68 |      9941.77 |               6 |     26288.3 | Random_large    |
| Random_large_200_11   | True       |             13 |             200 | []       | []                  | []                    |         70105.1 |           149.006  |          6496.23 |     14473.9  |              13 |     15374.5 | Random_large    |
| Clustered_large_200_6 | True       |              9 |             200 | []       | []                  | []                    |         26098.8 |            56.0494 |          5887.06 |     27704.6  |               9 |     14484.7 | Clustered_large |
| Clustered_large_200_5 | True       |              9 |             200 | []       | []                  | []                    |         24943.2 |            50.0988 |          5128.28 |     28204.9  |               9 |     14331   | Clustered_large |
| Mixed_large_200_18    | True       |             10 |             200 | []       | []                  | []                    |         58327.4 |           112.02   |          6705.66 |     11987.5  |              10 |     13080.3 | Mixed_large     |

## Best and worst per family
| family          | best_instance         |   best_objective |   best_routes | worst_instance        |   worst_objective |   worst_routes |
|:----------------|:----------------------|-----------------:|--------------:|:----------------------|------------------:|---------------:|
| Clustered_large | Clustered_large_200_4 |          9520.65 |             7 | Clustered_large_200_6 |          14484.7  |              9 |
| Clustered_tight | Clustered_tight_200_3 |          9136.44 |            20 | Clustered_tight_200_2 |           9909.45 |             22 |
| Random_large    | Random_large_200_10   |          6117.98 |             6 | Random_large_200_12   |          26288.3  |              6 |
| Random_tight    | Random_tight_200_8    |          5119.63 |            19 | Random_tight_200_7    |           6828.53 |             26 |
| Mixed_large     | Mixed_large_200_17    |         10221.2  |             8 | Mixed_large_200_18    |          13080.3  |             10 |
| Mixed_tight     | Mixed_tight_200_13    |          4448.36 |            19 | Mixed_tight_200_14    |           4676.73 |             19 |

## Correlation matrix
| metric           |   total_routes |   num_customers |   load_variance |   spatial_variance |   total_distance |   total_time |   vehicles_used |   objective |
|:-----------------|---------------:|----------------:|----------------:|-------------------:|-----------------:|-------------:|----------------:|------------:|
| total_routes     |       1        |             nan |       -0.523382 |          -0.527146 |        -0.163749 |    -0.208226 |        1        |   -0.583462 |
| num_customers    |     nan        |             nan |      nan        |         nan        |       nan        |   nan        |      nan        |  nan        |
| load_variance    |      -0.523382 |             nan |        1        |           0.998498 |         0.390656 |    -0.14894  |       -0.523382 |    0.906119 |
| spatial_variance |      -0.527146 |             nan |        0.998498 |           1        |         0.400999 |    -0.1592   |       -0.527146 |    0.900489 |
| total_distance   |      -0.163749 |             nan |        0.390656 |           0.400999 |         1        |    -0.409508 |       -0.163749 |    0.2405   |
| total_time       |      -0.208226 |             nan |       -0.14894  |          -0.1592   |        -0.409508 |     1        |       -0.208226 |    0.280807 |
| vehicles_used    |       1        |             nan |       -0.523382 |          -0.527146 |        -0.163749 |    -0.208226 |        1        |   -0.583462 |
| objective        |      -0.583462 |             nan |        0.906119 |           0.900489 |         0.2405   |     0.280807 |       -0.583462 |    1        |
