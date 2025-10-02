# Excel Insights - Workflow Guide

## Comparison Workflow Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                      EXCEL INSIGHTS                              │
│                    Comparison Workflow                           │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│  STEP 1: CHOOSE MODE                                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌──────────────────────┐      ┌──────────────────────┐        │
│  │  Two Files           │      │  Multiple Files      │        │
│  │  (Standard)          │      │  (3+ Files)          │        │
│  │                      │      │                      │        │
│  │  • Quick comparison  │      │  • Version tracking  │        │
│  │  • Side-by-side view │      │  • Change history    │        │
│  │  • Simple reports    │      │  • Timeline analysis │        │
│  └──────────────────────┘      └──────────────────────┘        │
│           │                               │                      │
│           └───────────┬───────────────────┘                      │
└───────────────────────┼──────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 2: UPLOAD FILES                                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  Standard Mode:                 Multi-File Mode:                 │
│  📄 File 1 (required)           📄 File 1 (oldest)              │
│  📄 File 2 (optional)           📄 File 2                        │
│                                  📄 File 3                        │
│                                  📄 File N (newest)              │
│                                                                   │
│  Supported formats: .xlsx, .xls                                  │
│                                                                   │
└───────────────────────┬──────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 3: SELECT SHEETS                                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  • Choose which sheet to compare                                 │
│  • Sheet must exist in all files (multi-file mode)              │
│  • Common sheets highlighted automatically                       │
│                                                                   │
└───────────────────────┬──────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 4: CONFIGURE COMPARISON                                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  Comparison Method:                                              │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────┐         │
│  │   Exact     │  │   Fuzzy     │  │  Numeric        │         │
│  │             │  │             │  │  Threshold      │         │
│  │ • Default   │  │ • 80% match │  │ • ±0.001        │         │
│  │ • Precise   │  │ • Text vars │  │ • Numbers       │         │
│  └─────────────┘  └─────────────┘  └─────────────────┘         │
│                                                                   │
│  Highlighting:                                                    │
│  🟡 Differences    🟢 Matches                                    │
│                                                                   │
└───────────────────────┬──────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 5: RUN COMPARISON                                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  Standard Mode:                 Multi-File Mode:                 │
│  📊 Compare Files              🔄 Compare All Versions           │
│                                                                   │
│  Processing:                                                      │
│  • Align dataframes                                              │
│  • Find common columns                                           │
│  • Perform cell-by-cell comparison                              │
│  • Track changes (multi-file)                                   │
│  • Generate statistics                                           │
│                                                                   │
└───────────────────────┬──────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 6: VIEW RESULTS                                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  Standard Mode Tabs:              Multi-File Mode Tabs:          │
│  ┌──────────────────┐            ┌──────────────────┐          │
│  │ 📊 Visual        │            │ 🔄 Timeline      │          │
│  │    Comparison    │            │    View          │          │
│  ├──────────────────┤            ├──────────────────┤          │
│  │ 📋 Detailed      │            │ 📋 Pairwise      │          │
│  │    Results       │            │    Comparisons   │          │
│  ├──────────────────┤            ├──────────────────┤          │
│  │ 📈 Statistics    │            │ 🎯 Change        │          │
│  │                  │            │    Tracking      │          │
│  ├──────────────────┤            ├──────────────────┤          │
│  │ 🔍 Differences   │            │ 📈 Statistics    │          │
│  │    & Matches     │            │                  │          │
│  └──────────────────┘            └──────────────────┘          │
│                                                                   │
└───────────────────────┬──────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 7: EXPORT RESULTS                                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  Export Options:                                                 │
│  • 📥 Excel Report (multiple sheets)                            │
│  • 📋 Summary Statistics                                         │
│  • 📊 Detailed Differences                                       │
│  • ✅ Match Tables                                               │
│  • 🔄 Complete Change History (multi-file)                      │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
```

## Data Flow Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                       DATA FLOW                                   │
└──────────────────────────────────────────────────────────────────┘

Excel Files (.xlsx, .xls)
         │
         ▼
┌─────────────────────┐
│  ExcelProcessor     │  • Load files
│                     │  • Clean data
│                     │  • Validate sheets
└──────────┬──────────┘
           │
           ▼
    DataFrame Objects
           │
           ▼
┌─────────────────────┐
│  ComparisonEngine   │  • Align dataframes
│                     │  • Compare cells
│                     │  • Track changes
│                     │  • Generate stats
└──────────┬──────────┘
           │
           ▼
   Comparison Results
           │
           ├──► Differences DataFrame
           ├──► Matches DataFrame
           ├──► Change Tracking Dict
           └──► Summary Statistics
           │
           ▼
┌─────────────────────┐
│ VisualizationEngine │  • Display results
│                     │  • Create charts
│                     │  • Format tables
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Streamlit UI       │  • Interactive display
│                     │  • User controls
│                     │  • Export options
└─────────────────────┘
```

## Key Algorithms

### Standard Comparison (2 Files)

```
1. Load Files
   └─► Parse Excel → DataFrame

2. Align DataFrames
   ├─► Find common columns
   ├─► Pad shorter DataFrame with NaN
   └─► Reorder columns to match

3. Cell-by-Cell Comparison
   For each cell (row, col):
   ├─► Get value from File 1
   ├─► Get value from File 2
   ├─► Apply comparison method:
   │   ├─► Exact: val1 == val2
   │   ├─► Fuzzy: similarity >= 80%
   │   └─► Numeric: |val1 - val2| <= threshold
   └─► Store result (True/False)

4. Generate Results
   ├─► Extract differences (False values)
   ├─► Extract matches (True values)
   ├─► Calculate statistics
   └─► Create highlight matrices
```

### Multi-File Comparison (3+ Files)

```
1. Load All Files
   └─► Parse each Excel → DataFrame[]

2. Find Common Structure
   ├─► Identify common columns across ALL files
   ├─► Find max row count
   └─► Align all DataFrames

3. Pairwise Comparisons
   For each consecutive pair (i, i+1):
   └─► Run standard comparison
   
4. Change Tracking
   For each cell (row, col):
   ├─► Collect values from all versions
   ├─► Compare consecutive versions
   ├─► Detect if changed
   └─► Store complete history

5. Aggregate Statistics
   ├─► Count total changed cells
   ├─► Calculate column change frequency
   ├─► Generate timeline data
   └─► Identify most volatile columns
```

## Performance Considerations

```
File Size Impact:
┌────────────┬───────────┬──────────────┐
│ Size       │ Time      │ Memory       │
├────────────┼───────────┼──────────────┤
│ < 1 MB     │ < 5s      │ Low          │
│ 1-5 MB     │ 5-15s     │ Medium       │
│ 5-10 MB    │ 15-30s    │ Medium-High  │
│ > 10 MB    │ 30s+      │ High         │
└────────────┴───────────┴──────────────┘

Row/Column Impact:
┌────────────┬──────────┬──────────────┐
│ Dimensions │ Time     │ Comparison   │
├────────────┼──────────┼──────────────┤
│ 100x10     │ Instant  │ Excellent    │
│ 1,000x50   │ < 2s     │ Good         │
│ 10,000x100 │ 5-10s    │ Acceptable   │
│ 50,000+    │ 15s+     │ Slow         │
└────────────┴──────────┴──────────────┘
```

## Troubleshooting Flowchart

```
Issue Detected
      │
      ▼
┌────────────────┐
│ No common      │
│ columns?       │
└─────┬──────────┘
      │ Yes
      ▼
Check column names
in all files
      │
      ▼
Rename columns to
match exactly
      
┌────────────────┐
│ Shape          │
│ mismatch?      │
└─────┬──────────┘
      │ Yes
      ▼
Tool will auto-pad
with NaN values
(No action needed)

┌────────────────┐
│ Sheet not      │
│ found?         │
└─────┬──────────┘
      │ Yes
      ▼
Verify sheet names
Select correct sheet

┌────────────────┐
│ Slow           │
│ performance?   │
└─────┬──────────┘
      │ Yes
      ▼
Reduce file size
or row count
```

---

For more detailed information, see [FEATURES.md](FEATURES.md)
