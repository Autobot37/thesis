# City x Season Analysis

## 1. EDA: Temperature Statistics
| city      | season       |    mean |      var |   min |   max |
|:----------|:-------------|--------:|---------:|------:|------:|
| Ahmedabad | Monsoon      | 32.784  | 12.8625  |  24.8 |  43.8 |
| Ahmedabad | Post-Monsoon | 32.9694 |  4.73251 |  24.4 |  39.8 |
| Ahmedabad | Summer       | 38.1087 | 10.7528  |  21.5 |  46.4 |
| Ahmedabad | Winter       | 28.714  |  6.8191  |  19.5 |  36.9 |
| Bangalore | Monsoon      | 26.8957 |  2.70465 |  20.8 |  35.4 |
| Bangalore | Post-Monsoon | 26.2946 |  2.93256 |  18.3 |  31.6 |
| Bangalore | Summer       | 32.2018 |  4.62358 |  22.4 |  39.4 |
| Bangalore | Winter       | 27.6445 |  4.53978 |  18.5 |  33.9 |
| Chennai   | Monsoon      | 33.8259 |  4.90562 |  26   |  40   |
| Chennai   | Post-Monsoon | 29.6527 |  4.16001 |  24   |  36.1 |
| Chennai   | Summer       | 34.2029 |  6.54506 |  24.6 |  41.9 |
| Chennai   | Winter       | 28.4565 |  2.01553 |  23.1 |  34.2 |
| Delhi     | Monsoon      | 34.2717 | 13.4931  |  24.3 |  45.4 |
| Delhi     | Post-Monsoon | 29.6822 | 10.3891  |  20.4 |  37.4 |
| Delhi     | Summer       | 35.2262 | 28.7891  |  18.1 |  46.4 |
| Delhi     | Winter       | 21.7114 | 10.5614  |  12.3 |  32.5 |
| Hyderabad | Monsoon      | 29.7688 |  8.33448 |  22.3 |  42.2 |
| Hyderabad | Post-Monsoon | 29.2089 |  3.06305 |  20.3 |  34.3 |
| Hyderabad | Summer       | 36.3269 |  6.57755 |  22.3 |  41.8 |
| Hyderabad | Winter       | 29.3767 |  5.26722 |  21   |  36.6 |
| Jaipur    | Monsoon      | 33.5931 | 14.7995  |  23.8 |  45   |
| Jaipur    | Post-Monsoon | 30.2782 | 11.6654  |  16.8 |  39   |
| Jaipur    | Summer       | 36.1484 | 23.2239  |  20.2 |  46.4 |
| Jaipur    | Winter       | 23.2019 | 11.1471  |  15   |  33.4 |
| Kolkata   | Monsoon      | 31.5235 |  4.24774 |  25.1 |  41.4 |
| Kolkata   | Post-Monsoon | 29.4548 |  3.32506 |  22   |  34.2 |
| Kolkata   | Summer       | 34.5443 |  7.30917 |  23.6 |  43.9 |
| Kolkata   | Winter       | 26.45   |  7.27601 |  17.7 |  36.6 |
| Lucknow   | Monsoon      | 33.2484 | 12.3879  |  24.2 |  45.7 |
| Lucknow   | Post-Monsoon | 29.5315 |  7.14047 |  21   |  37.3 |
| Lucknow   | Summer       | 35.966  | 20.2614  |  20.3 |  46.3 |
| Lucknow   | Winter       | 22.9576 |  9.72005 |  12.5 |  35   |
| Mumbai    | Monsoon      | 28.5242 |  2.47878 |  25.4 |  34.9 |
| Mumbai    | Post-Monsoon | 31.71   |  2.65928 |  25.1 |  35.9 |
| Mumbai    | Summer       | 32.6408 |  3.27668 |  24.9 |  41.4 |
| Mumbai    | Winter       | 30.2242 |  4.127   |  22.1 |  36.3 |
| Pune      | Monsoon      | 27.5079 |  6.65009 |  22.1 |  39.2 |
| Pune      | Post-Monsoon | 29.7889 |  2.67523 |  22.3 |  34.5 |
| Pune      | Summer       | 36.201  |  6.0047  |  23.2 |  42.9 |
| Pune      | Winter       | 29.991  |  4.65776 |  21.2 |  36.6 |

## 2. Model: DLinear (No City Embeddings) MSE
| city      | season       |      mse |
|:----------|:-------------|---------:|
| Ahmedabad | Monsoon      |  5.96475 |
| Ahmedabad | Post-Monsoon |  4.27878 |
| Ahmedabad | Summer       |  7.60736 |
| Ahmedabad | Winter       |  5.4869  |
| Bangalore | Monsoon      |  2.51076 |
| Bangalore | Post-Monsoon |  2.95371 |
| Bangalore | Summer       |  5.27635 |
| Bangalore | Winter       |  2.06784 |
| Chennai   | Monsoon      |  4.8401  |
| Chennai   | Post-Monsoon |  3.57517 |
| Chennai   | Summer       |  4.99807 |
| Chennai   | Winter       |  1.02401 |
| Delhi     | Monsoon      |  8.22875 |
| Delhi     | Post-Monsoon |  6.23511 |
| Delhi     | Summer       | 16.4196  |
| Delhi     | Winter       |  8.20748 |
| Hyderabad | Monsoon      |  4.56565 |
| Hyderabad | Post-Monsoon |  1.66282 |
| Hyderabad | Summer       |  6.72874 |
| Hyderabad | Winter       |  2.29453 |
| Jaipur    | Monsoon      |  4.85336 |
| Jaipur    | Post-Monsoon |  6.57838 |
| Jaipur    | Summer       | 12.9007  |
| Jaipur    | Winter       |  7.42013 |
| Kolkata   | Monsoon      |  3.39303 |
| Kolkata   | Post-Monsoon |  2.94474 |
| Kolkata   | Summer       | 11.6765  |
| Kolkata   | Winter       |  5.29439 |
| Lucknow   | Monsoon      |  6.64104 |
| Lucknow   | Post-Monsoon |  3.77186 |
| Lucknow   | Summer       | 14.8686  |
| Lucknow   | Winter       |  7.15858 |
| Mumbai    | Monsoon      |  1.81157 |
| Mumbai    | Post-Monsoon |  2.73929 |
| Mumbai    | Summer       |  4.86609 |
| Mumbai    | Winter       |  5.20902 |
| Pune      | Monsoon      |  4.15625 |
| Pune      | Post-Monsoon |  1.99366 |
| Pune      | Summer       |  5.38205 |
| Pune      | Winter       |  3.03271 |