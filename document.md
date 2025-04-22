**user_message:**

I want to do an enterprise valuation of Walmart as of the end of 2010.  I will start from Walmart's income statements and balance sheets for 2009 and 2010.  I want to revise those statements to calculate NOPAT instead of net income and to calculate Invested Capital.  When I am finished with enterprise valuation, I will want to compute the equity value.  Can you help with that if I upload the past statements?

**ai_message:**

I'll help you perform an enterprise valuation of Walmart using their 2009-2010 financial statements. This will involve several steps:

1. First, we'll need to:
- Adjust the income statement to calculate NOPAT (Net Operating Profit After Taxes)
- Reorganize the balance sheet to calculate Invested Capital
- Remove non-operating items
- Make necessary tax adjustments

2. Then we can:
- Calculate the free cash flows
- Determine an appropriate growth rate and WACC
- Perform DCF analysis to get enterprise value
- Adjust for debt and cash to arrive at equity value

Please upload Walmart's:
- Income statements for 2009-2010
- Balance sheets for 2009-2010

Once you upload these statements, we can start with the adjustments to calculate NOPAT and Invested Capital. The key adjustments we'll make include:

For NOPAT:
- Adding back interest expense (net of tax shield)
- Removing non-operating income/expenses
- Adjusting for operating leases if present
- Normalizing tax rates

For Invested Capital:
- Net working capital
- Net PP&E
- Other operating assets
- Capitalizing operating leases if present
- Removing non-operating assets

Please upload the financial statements and we can begin the analysis step by step.

**user_message:**

I uploaded the file: `WalMart.xlsx`

The income statements are in the worksheet 1a Income Statements and the balance sheets are in 1b Balance Sheets.  There is a blank line between Assets and Liabilities in the balance sheet worksheet, so be sure you get both.  As you make the adjustments to calculate NOPAT and Invested Capital, explain each adjustment you plan to make and get approval before continuing.

**ai_message:**

I'll start by loading and examining the financial statements to ensure we have the correct data.


```
# Load the Excel file
import pandas as pd

# Read both sheets
dataframes = read_all_sheets_from_excel('WalMart.xlsx')

# Display Income Statement
print("Income Statement:")
print(dataframes['1a Income Statements'])
print("\nBalance Sheet:")
print(dataframes['1b Balance Sheets'])
```

**ai_message:**

Let me try reading the Excel file differently:


```
# Load the Excel file
import pandas as pd

# Read both sheets
income_stmt = pd.read_excel('WalMart.xlsx', sheet_name='1a Income Statements')
balance_sheet = pd.read_excel('WalMart.xlsx', sheet_name='1b Balance Sheets')

print("Income Statement:")
print(income_stmt)
print("\nBalance Sheet:")
print(balance_sheet)
```

**ai_message:**

Let me help you analyze the financial statements. First, let's look at what we have and then outline the adjustments we'll need to make to calculate NOPAT and Invested Capital.

{outputs_dict['e947f968']}
{outputs_dict['ff772101']}

Based on the financial statements, here are the adjustments I propose to calculate NOPAT and Invested Capital:

For NOPAT Calculation:
1. Start with Operating Income (before interest and taxes):
   - Net Sales + Other Income - COGS - Operating Expenses
2. Add back:
   - Interest expense (net) since this will be considered in WACC
3. Remove:
   - Earnings from discontinued operations (non-operating)
4. Apply adjusted tax rate:
   - Will use the effective tax rate from the income statement

For Invested Capital Calculation:
1. Operating Working Capital:
   - Current Assets (excluding cash and discontinued operations)
   - Less: Non-interest-bearing current liabilities (accounts payable and accrued liabilities)
2. Long-term Operating Assets:
   - Net PPE
   - Property under capital leases
   - Goodwill (since it represents operating acquisitions)
3. Other operating assets

Would you like me to proceed with these adjustments? Also, please confirm if there are any specific adjustments you'd like me to add or modify before we calculate NOPAT and Invested Capital.

**user_message:**

Exclude other income from NOPAT.  Also, exclude other income from the effective tax calculation, because it is shown as net income.  Exclude other assets from the invested capital calculation.  What do you plan to do with minority interests and with the obligations under capital leases on the balance sheet?  Please state your revised plan and get approval before continuing.

**ai_message:**

Thank you for the clarifications. Here's my revised plan:

For NOPAT Calculation:
1. Start with Operating Income:
   - Net Sales (excluding Other Income)
   - Less: COGS
   - Less: Operating, selling, and general administrative expenses
2. Calculate Adjusted Tax Rate:
   - Remove impact of other income and interest expense from tax calculation
   - Tax Rate = Income Tax / (Earnings before tax - Other Income + Interest Expense)
3. Apply adjusted tax rate to operating income
4. Deduct Minority Interest (as it represents operating claims of minority shareholders)

For Invested Capital Calculation:
1. Operating Working Capital:
   - Current Assets (excluding cash and discontinued operations)
   - Less: Non-interest-bearing current liabilities (accounts payable and accrued liabilities)
2. Long-term Operating Assets:
   - Net PPE
   - Property under capital leases
   - Add: Obligations under capital leases (since this is operating financing)
   - Goodwill
3. Exclude:
   - Other assets
   - Other income
   - Cash and cash equivalents
   - Discontinued operations

The treatment of capital leases: Since these are operating assets, I plan to include both the asset value and the corresponding lease obligations in invested capital, as they represent operating assets and their financing.

Would you like me to proceed with these adjustments? Please let me know if you'd like any modifications to this revised approach.

**user_message:**

Include capital leases in assets.  But exclude the obligations from liabilities, because they are really debt items.  We will subtract them along with debt when we derive equity value from enterprise value.  Please restate your plan again and get approval before continuing.

**ai_message:**

Thank you for the clarification. Here's my final revised plan:

For NOPAT Calculation:
1. Start with Operating Income:
   - Net Sales (excluding Other Income)
   - Less: COGS
   - Less: Operating, selling, and general administrative expenses
2. Calculate Adjusted Tax Rate:
   - Remove impact of other income and interest expense from tax calculation
   - Tax Rate = Income Tax / (Earnings before tax - Other Income + Interest Expense)
3. Apply adjusted tax rate to operating income
4. Deduct Minority Interest (as it represents operating claims of minority shareholders)

For Invested Capital Calculation:
1. Operating Working Capital:
   - Current Assets (excluding cash and discontinued operations)
   - Less: Non-interest-bearing current liabilities (accounts payable and accrued liabilities)
2. Long-term Operating Assets:
   - Net PPE
   - Property under capital leases (included as operating assets)
   - Goodwill
3. Exclude:
   - Other assets
   - Other income
   - Cash and cash equivalents
   - Discontinued operations

For Final Equity Value Calculation:
Will subtract from Enterprise Value:
- Short-term borrowings
- Long-term debt
- Obligations under capital leases
- Add back: Cash and cash equivalents

Would you like me to proceed with these calculations? Please let me know if you'd like any modifications to this final approach.

**user_message:**

do not deduct minority interest from NOPAT.  We will subtract minority interests from enterprise value to get equity value.  Please restate the plan for approval again.

**ai_message:**

Here's the final plan:

For NOPAT Calculation:
1. Start with Operating Income:
   - Net Sales (excluding Other Income)
   - Less: COGS
   - Less: Operating, selling, and general administrative expenses
2. Calculate Adjusted Tax Rate:
   - Remove impact of other income and interest expense from tax calculation
   - Tax Rate = Income Tax / (Earnings before tax - Other Income + Interest Expense)
3. Apply adjusted tax rate to operating income
   (No deduction of minority interest from NOPAT)

For Invested Capital Calculation:
1. Operating Working Capital:
   - Current Assets (excluding cash and discontinued operations)
   - Less: Non-interest-bearing current liabilities (accounts payable and accrued liabilities)
2. Long-term Operating Assets:
   - Net PPE
   - Property under capital leases (included as operating assets)
   - Goodwill
3. Exclude:
   - Other assets
   - Other income
   - Cash and cash equivalents
   - Discontinued operations

For Final Equity Value Calculation:
Will subtract from Enterprise Value:
- Short-term borrowings
- Long-term debt
- Obligations under capital leases
- Minority interests
+ Add back: Cash and cash equivalents

Would you like me to proceed with these calculations? Please let me know if you'd like any modifications to this final approach.

**user_message:**

the effective tax rate should be calculated as taxes / (income before tax - other income).  please revise the plan again.

**ai_message:**

Here's the final revised plan:

For NOPAT Calculation:
1. Start with Operating Income:
   - Net Sales (excluding Other Income)
   - Less: COGS
   - Less: Operating, selling, and general administrative expenses
2. Calculate Adjusted Tax Rate:
   - Effective Tax Rate = Income Tax / (Earnings before tax - Other Income)
   - Apply this tax rate to operating income
   (No deduction of minority interest from NOPAT)

For Invested Capital Calculation:
1. Operating Working Capital:
   - Current Assets (excluding cash and discontinued operations)
   - Less: Non-interest-bearing current liabilities (accounts payable and accrued liabilities)
2. Long-term Operating Assets:
   - Net PPE
   - Property under capital leases (included as operating assets)
   - Goodwill
3. Exclude:
   - Other assets
   - Other income
   - Cash and cash equivalents
   - Discontinued operations

For Final Equity Value Calculation:
Will subtract from Enterprise Value:
- Short-term borrowings
- Long-term debt
- Obligations under capital leases
- Minority interests
+ Add back: Cash and cash equivalents

Would you like me to proceed with these calculations? Please let me know if you'd like any modifications to this final approach.

**user_message:**

yes, please proceed with that plan.  save the NOPAT calculation in a worksheet of an excel file and save the Invested Capital calculation in a separate worksheet of the same file and provide a download link.

**ai_message:**

I'll calculate NOPAT and Invested Capital and save them in separate worksheets.


```
import pandas as pd
import numpy as np

# First calculate NOPAT
# Start with operating revenues and expenses
operating_data = {
    'Item': [
        'Net Sales',
        'Cost of Goods Sold',
        'Operating, Selling, and G&A Expenses',
        'Operating Income',
        '',
        'Tax Rate Calculation:',
        'Income Before Tax',
        'Less: Other Income',
        'Adjusted Income Before Tax',
        'Income Tax',
        'Effective Tax Rate',
        '',
        'NOPAT Calculation:',
        'Operating Income',
        'Less: Taxes on Operating Income',
        'NOPAT'
    ],
    '2009': [
        401087,  # Net sales
        304056,  # COGS
        77520,   # Operating expenses
        401087 - 304056 - 77520,  # Operating income
        None,
        None,
        20898,   # Income before tax
        3287,    # Other income
        20898 - 3287,  # Adjusted income before tax
        7145,    # Income tax
        7145 / (20898 - 3287),  # Effective tax rate
        None,
        None,
        401087 - 304056 - 77520,  # Operating income
        (401087 - 304056 - 77520) * (7145 / (20898 - 3287)),  # Taxes on operating income
        (401087 - 304056 - 77520) * (1 - 7145 / (20898 - 3287))  # NOPAT
    ],
    '2010': [
        405046,  # Net sales
        304657,  # COGS
        79607,   # Operating expenses
        405046 - 304657 - 79607,  # Operating income
        None,
        None,
        22066,   # Income before tax
        3168,    # Other income
        22066 - 3168,  # Adjusted income before tax
        7139,    # Income tax
        7139 / (22066 - 3168),  # Effective tax rate
        None,
        None,
        405046 - 304657 - 79607,  # Operating income
        (405046 - 304657 - 79607) * (7139 / (22066 - 3168)),  # Taxes on operating income
        (405046 - 304657 - 79607) * (1 - 7139 / (22066 - 3168))  # NOPAT
    ]
}

nopat_df = pd.DataFrame(operating_data)

# Now calculate Invested Capital
invested_capital_data = {
    'Item': [
        'Operating Working Capital:',
        'Current Assets (excl. cash and disc. ops)',
        'Less: Non-interest Current Liabilities',
        'Net Operating Working Capital',
        '',
        'Long-term Operating Assets:',
        'Net PPE',
        'Property under Capital Leases',
        'Goodwill',
        '',
        'Total Invested Capital'
    ],
    '2009': [
        None,
        34511 + 3905 + 3063,  # Inventory + AR + Prepaid (excl. cash and disc. ops)
        28849 + 18789,  # AP + Accrued liabilities
        (34511 + 3905 + 3063) - (28849 + 18789),
        None,
        None,
        92856,  # Net PPE
        2797,   # Property under capital leases
        15260,  # Goodwill
        None,
        (34511 + 3905 + 3063) - (28849 + 18789) + 92856 + 2797 + 15260
    ],
    '2010': [
        None,
        33160 + 4144 + 2980,  # Inventory + AR + Prepaid (excl. cash and disc. ops)
        30451 + 20099,  # AP + Accrued liabilities
        (33160 + 4144 + 2980) - (30451 + 20099),
        None,
        None,
        99544,  # Net PPE
        2763,   # Property under capital leases
        16126,  # Goodwill
        None,
        (33160 + 4144 + 2980) - (30451 + 20099) + 99544 + 2763 + 16126
    ]
}

invested_capital_df = pd.DataFrame(invested_capital_data)

# Create Excel writer object
with pd.ExcelWriter('Walmart_NOPAT_and_IC.xlsx') as writer:
    nopat_df.to_excel(writer, sheet_name='NOPAT', index=False)
    invested_capital_df.to_excel(writer, sheet_name='Invested Capital', index=False)

print("NOPAT Calculation:")
print(nopat_df)
print("\nInvested Capital Calculation:")
print(invested_capital_df)
```

**ai_message:**

I've created the Excel file 'Walmart_NOPAT_and_IC.xlsx' with two worksheets containing the NOPAT and Invested Capital calculations. Here are the results:

{outputs_dict['d7159bc1']}

{outputs_dict['3f7f7017']}

Key findings:
- NOPAT increased from $11,595M in 2009 to $12,931M in 2010
- Invested Capital increased from $104,754M in 2009 to $108,167M in 2010
- The effective tax rate decreased from 40.57% in 2009 to 37.78% in 2010
- Walmart maintains negative operating working capital, which is typical for efficient retailers

The Excel file has been saved and can be downloaded at https://julius.ai/files/Walmart_NOPAT_and_IC.xlsx

Would you like to proceed with the enterprise valuation using these calculations?

**user_message:**

Explain each line of the NOPAT calculation.

**ai_message:**

I'll explain each line of the NOPAT calculation:

{outputs_dict['d7159bc1']}

1. Net Sales: Pure operating revenue
   - 2009: $401,087M
   - 2010: $405,046M
   - Note: Excludes other income as it's non-operating

2. Cost of Goods Sold: Direct costs of products sold
   - 2009: $304,056M
   - 2010: $304,657M

3. Operating, Selling, and G&A Expenses: Operating overhead
   - 2009: $77,520M
   - 2010: $79,607M

4. Operating Income: Net Sales - COGS - Operating Expenses
   - 2009: $401,087M - $304,056M - $77,520M = $19,511M
   - 2010: $405,046M - $304,657M - $79,607M = $20,782M

Tax Rate Calculation:
5. Income Before Tax: From income statement
   - 2009: $20,898M
   - 2010: $22,066M

6. Less: Other Income: Removed as non-operating
   - 2009: $3,287M
   - 2010: $3,168M

7. Adjusted Income Before Tax: Income Before Tax - Other Income
   - 2009: $20,898M - $3,287M = $17,611M
   - 2010: $22,066M - $3,168M = $18,898M

8. Income Tax: Actual taxes paid
   - 2009: $7,145M
   - 2010: $7,139M

9. Effective Tax Rate: Income Tax / Adjusted Income Before Tax
   - 2009: $7,145M / $17,611M = 40.57%
   - 2010: $7,139M / $18,898M = 37.78%

NOPAT Calculation:
10. Operating Income: Same as line 4
    - 2009: $19,511M
    - 2010: $20,782M

11. Less: Taxes on Operating Income: Operating Income × Effective Tax Rate
    - 2009: $19,511M × 40.57% = $7,916M
    - 2010: $20,782M × 37.78% = $7,851M

12. NOPAT: Operating Income - Taxes on Operating Income
    - 2009: $19,511M - $7,916M = $11,595M
    - 2010: $20,782M - $7,851M = $12,931M

This shows how we:
1. Started with pure operating performance (excluding other income)
2. Calculated a tax rate based on operating income (by removing other income from the denominator)
3. Applied this tax rate to operating income to get NOPAT
4. Did not deduct minority interest (as agreed, this will be handled in enterprise to equity value conversion)

**user_message:**

Explain each line of the Invested Capital calculation.

**ai_message:**

I'll explain each line of the Invested Capital calculation:

{outputs_dict['3f7f7017']}

Operating Working Capital:
1. Current Assets (excl. cash and disc. ops):
   - 2009: $41,479M = Inventory ($34,511M) + Accounts Receivable ($3,905M) + Prepaid Expenses ($3,063M)
   - 2010: $40,284M = Inventory ($33,160M) + Accounts Receivable ($4,144M) + Prepaid Expenses ($2,980M)
   - Note: Excludes cash and discontinued operations as non-operating

2. Less: Non-interest Current Liabilities:
   - 2009: $47,638M = Accounts Payable ($28,849M) + Accrued Liabilities ($18,789M)
   - 2010: $50,550M = Accounts Payable ($30,451M) + Accrued Liabilities ($20,099M)
   - Note: Excludes short-term debt as it's financing, not operating

3. Net Operating Working Capital: Current Assets - Non-interest Current Liabilities
   - 2009: $41,479M - $47,638M = -$6,159M
   - 2010: $40,284M - $50,550M = -$10,266M
   - Note: Negative working capital is common in retail due to inventory turnover and supplier financing

Long-term Operating Assets:
4. Net PPE: Property, Plant & Equipment net of depreciation
   - 2009: $92,856M
   - 2010: $99,544M

5. Property under Capital Leases: Operating assets financed through leases
   - 2009: $2,797M
   - 2010: $2,763M
   - Note: Included as operating assets while lease obligations will be treated as debt

6. Goodwill: From acquisitions
   - 2009: $15,260M
   - 2010: $16,126M

7. Total Invested Capital: Sum of Net Operating Working Capital and Long-term Operating Assets
   - 2009: -$6,159M + $92,856M + $2,797M + $15,260M = $104,754M
   - 2010: -$10,266M + $99,544M + $2,763M + $16,126M = $108,167M

Key observations:
- Negative working capital reduces the total invested capital needed
- PPE is the largest component, reflecting Walmart's extensive store network
- Goodwill shows significant past acquisition activity
- Capital leases are included in assets but their financing will be treated as debt when converting enterprise to equity value
- Other assets were excluded as non-operating per our agreed approach
