# Excel Insights

A powerful tool to analyze and compare Excel files, highlighting differences and tracking changes over time - faster and more intuitive than traditional Excel formulas.

## 🚀 Key Features

### Standard Comparison (2 Files)
- **Side-by-side comparison** of two Excel files
- **Color-coded highlighting** of differences and matches
- **Detailed difference reports** showing exactly what changed
- **Multiple comparison methods**: exact, fuzzy, and numeric threshold
- **Natural language queries** for intuitive analysis

### Multi-File Version Tracking (3+ Files) 🆕
- **Track changes across multiple versions** of Excel files over time
- **Cell-level change history** showing how each value evolved
- **Timeline visualization** of match percentages across versions
- **Identify most frequently changed columns** and patterns
- **Comprehensive change reports** for auditing and analysis

### Smart Features
- ✅ Detects fields present in one file but not the other
- ✅ Handles numeric differences with configurable thresholds
- ✅ Text matching with fuzzy comparison for variations
- ✅ Missing value detection and handling
- ✅ Export results to Excel with detailed breakdowns
- ✅ Visual charts and statistics

## 📖 Quick Start

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the application:**
   ```bash
   streamlit run app.py
   ```

3. **Upload your files:**
   - Choose "Two Files" mode for standard comparison
   - Choose "Multiple Files (3+)" mode to track changes over time

4. **Analyze:**
   - View highlighted differences
   - Explore change history
   - Export comprehensive reports

## 💡 Use Cases

- **Version Control**: Track how data changes across monthly/quarterly updates
- **Data Validation**: Compare source and target files to identify discrepancies
- **Audit Trails**: Monitor modifications to sensitive data over time
- **Quality Assurance**: Validate data transformations and migrations
- **Price Tracking**: Monitor product price changes across catalogs
- **Inventory Management**: Track stock level changes over time

## 📚 Documentation

For detailed feature documentation, see [FEATURES.md](FEATURES.md)

## 🛠️ Technology Stack

- **pandas**: Data manipulation and analysis
- **openpyxl**: Excel file processing
- **streamlit**: Interactive web interface
- **plotly**: Beautiful visualizations and charts

## 🎯 What Makes It Better Than Formulas?

- **No formula writing required** - intuitive point-and-click interface
- **Visual highlighting** - instantly see what changed
- **Multi-file tracking** - Excel formulas can't track changes across 3+ files
- **Comprehensive reports** - automatic generation of detailed analysis
- **Timeline views** - understand change patterns over time
- **Natural language** - ask questions in plain English

## 📊 Example

Compare three monthly product catalogs:
1. Upload `products_jan.xlsx`, `products_feb.xlsx`, `products_mar.xlsx`
2. Click "Compare All Versions"
3. See exactly which products changed, when, and how
4. Export a complete change report

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is open source and available under the MIT License.
