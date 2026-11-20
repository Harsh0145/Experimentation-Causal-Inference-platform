# CSV upload demo

Upload `demo_ab_upload.csv` in the app sidebar, then open **Experiment analysis**. The app should detect `variant` as the assignment column and `converted` as a binary outcome.

## Expected result

- 500 users per arm
- Control: 50 conversions, or 10%
- Treatment: 100 conversions, or 20%
- Treatment effect: +10 percentage points
- Two-proportion z-test: z ≈ 4.428, p ≈ 0.0000095
- 95% confidence interval shown by the app: approximately +5.62 to +14.38 percentage points

At the default 5% significance level, the app should report a statistically significant difference in conversion. These rows are synthetic and are intended only to check the CSV upload and analysis flow.
