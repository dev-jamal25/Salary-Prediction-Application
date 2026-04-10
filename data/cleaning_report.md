# Data Cleaning and EDA Report

## 1. Dataset Overview

**Original rows:** 607
**Final rows:** 565
**Columns:** 11

**Columns:** work_year, experience_level, employment_type, job_title, salary, salary_currency, salary_in_usd, employee_residence, remote_ratio, company_location, company_size


## 2. Data Quality Checks

| Check | Result |
|-------|--------|
| Null count | 0 |
| Duplicates | 0 |
| Whitespace issues | 0 found |


## 3. Data Types

| Column | Type | Sample |
|--------|------|--------|
| work_year | int64 | 2020 |
| experience_level | str | MI |
| employment_type | str | FT |
| job_title | str | Data Scientist |
| salary | int64 | 70000 |
| salary_currency | str | EUR |
| salary_in_usd | int64 | 79833 |
| employee_residence | str | DE |
| remote_ratio | int64 | 0 |
| company_location | str | DE |
| company_size | str | L |


## 4. Categorical Features Distribution


### experience_level
| Value | Count |
|-------|-------|
| EN | 88 |
| EX | 26 |
| MI | 208 |
| SE | 243 |

### employment_type
| Value | Count |
|-------|-------|
| CT | 5 |
| FL | 4 |
| FT | 546 |
| PT | 10 |

### company_size
| Value | Count |
|-------|-------|
| L | 193 |
| M | 290 |
| S | 82 |

### remote_ratio
| Value | Count |
|-------|-------|
| 0 | 121 |
| 50 | 98 |
| 100 | 346 |

### job_title (Top 10)
| Title | Count |
|-------|-------|
| Data Scientist | 130 |
| Data Engineer | 121 |
| Data Analyst | 82 |
| Machine Learning Engineer | 39 |
| Research Scientist | 16 |
| Data Science Manager | 12 |
| Data Architect | 11 |
| Machine Learning Scientist | 8 |
| Big Data Engineer | 8 |
| Data Science Consultant | 7 |

### company_location (Top 10)
| Location | Count |
|----------|-------|
| US | 318 |
| GB | 46 |
| CA | 28 |
| DE | 27 |
| IN | 24 |
| FR | 15 |
| ES | 14 |
| GR | 10 |
| JP | 6 |
| NL | 4 |

### employee_residence (Top 10)
| Residence | Count |
|-----------|-------|
| US | 295 |
| GB | 43 |
| IN | 30 |
| CA | 27 |
| DE | 24 |
| FR | 18 |
| ES | 15 |
| GR | 12 |
| JP | 7 |
| PK | 6 |

## 5. Numeric Features - Sanity Checks

### work_year
- Min: 2020
- Max: 2022
- Unique values: [2020, 2021, 2022]


### remote_ratio
- Min: 0
- Max: 100
- Unique values: [0, 50, 100]


### salary_in_usd
- Min: $2,859.00
- Max: $600,000.00
- Mean: $110,610.34
- Median: $100,000.00
- Std: $72,280.70


## 6. Transformations Applied

1. Dropped accidental index column (Unnamed: 0)
2. Removed 42 duplicate rows
3. Trimmed leading/trailing whitespace from string columns

## 7. Warnings / Issues Found

- Unusually low salary: $2,859

## 8. Data Dictionary (Final)

| Feature | Type | Allowed Values / Range |
|---------|------|------------------------|
| work_year | numeric | [2020, 2021, 2022] |
| experience_level | categorical | ['EN', 'EX', 'MI', 'SE'] |
| employment_type | categorical | ['CT', 'FL', 'FT', 'PT'] |
| job_title | categorical | 50 unique values |
| salary | int64 | (excluded from model) |
| salary_currency | str | (excluded from model) |
| salary_in_usd | numeric | $2,859 - $600,000 |
| employee_residence | categorical | 57 unique values (ISO codes) |
| remote_ratio | numeric | [0, 50, 100] |
| company_location | categorical | 50 unique values (ISO codes) |
| company_size | categorical | ['L', 'M', 'S'] |