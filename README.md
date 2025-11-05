# NoSQL Database Assignment - Recipe Management System

**Course:** BCSE406L - NoSQL Database  
**Slot:** A1+TA1  
**Database Type:** Document Database (MongoDB)  
**Application:** Recipe Management System

## Project Overview

This project implements a **Recipe Management System** using MongoDB (Document Database) to demonstrate NoSQL database concepts including:

1. **Flexible Schema Design** - Recipes can have different fields based on their type
2. **CRUD Operations** - Complete Create, Read, Update, Delete functionality
3. **Embedded Documents** - Ingredients and instructions stored within recipe documents
4. **Array Operations** - Native support for tags, dietary info arrays
5. **Full-Text Search** - Search across recipe titles, descriptions, and ingredients
6. **Indexing** - Optimized indexes for fast query performance

## Key NoSQL Benefits Demonstrated

### Flexible Schema
- Different recipes can have different fields
- No rigid schema enforcement
- Easy to add new fields without migrations
- Example: Basic recipes have minimal fields, while advanced recipes include nutrition info, equipment lists, etc.

### Document Structure
- Embedded documents for related data (ingredients, instructions)
- No joins required for retrieving complete recipe information
- Faster queries and simpler data structure

### Native Array Support
- Tags, dietary information stored as arrays
- Powerful array query and update operations
- Efficient operations like `$addToSet`, `$in`, etc.

## Project Structure

```
.
├── database_setup.py       # Database connection and configuration
├── data_model.py          # Data model definitions
├── crud_operations.py     # CRUD operations implementation
├── demonstration.py       # Complete demonstration script
├── requirements.txt       # Python dependencies
├── README.md             # This file
└── ASSIGNMENT_REPORT.md  # Detailed assignment report
```

## Prerequisites

1. **Python 3.7+**
2. **MongoDB** - Install MongoDB on your system
   - Windows: Download from [MongoDB Download Center](https://www.mongodb.com/try/download/community)
   - Linux: `sudo apt-get install mongodb` or `sudo yum install mongodb`
   - Mac: `brew install mongodb-community`

## Installation

1. **Install Python Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Start MongoDB Service:**
   - **Windows:**
     ```bash
     net start MongoDB
     ```
   - **Linux/Mac:**
     ```bash
     sudo systemctl start mongod
     # or
     mongod --dbpath /path/to/data/directory
     ```

3. **Verify MongoDB is Running:**
   - Check if MongoDB is running on default port (27017)
   - You can verify by trying to connect: `mongosh` or `mongo`

## Usage

### 1. Database Setup
Run the database setup script to create collections and indexes:
```bash
python database_setup.py
```

### 2. Run Full Demonstration
Run the complete demonstration script:
```bash
python demonstration.py
```

This will demonstrate:
- Database setup and configuration
- Flexible schema examples
- All CRUD operations
- Embedded documents
- Array operations
- Performance features

## Features Implemented

### Database Setup and Configuration (5 marks)
- ✅ MongoDB connection handling
- ✅ Database and collection creation
- ✅ Index creation for optimal performance
- ✅ Error handling and connection validation
- ✅ Database statistics and monitoring

### Data Model Design (10 marks)
- ✅ Flexible schema design for recipes
- ✅ Embedded document structure (ingredients, instructions)
- ✅ Array fields (tags, dietary info)
- ✅ Optional fields based on recipe type
- ✅ Category and user models for extensibility

### CRUD Operations Implementation (10 marks)
- ✅ **CREATE**: Insert recipes and categories
- ✅ **READ**: 
  - Read by ID
  - Read all with pagination
  - Read by category
  - Read by difficulty
  - Full-text search
  - Read by tags
- ✅ **UPDATE**: 
  - Update recipe fields
  - Update ratings
  - Add tags using array operators
- ✅ **DELETE**: Remove recipes and categories

### Report and Demonstration (5 marks)
- ✅ Complete demonstration script
- ✅ Comprehensive documentation
- ✅ Code comments and explanations
- ✅ Assignment report (see `ASSIGNMENT_REPORT.md`)

## Code Examples

### Creating a Recipe
```python
from database_setup import DatabaseConfig
from data_model import RecipeModel
from crud_operations import RecipeCRUD

# Connect to database
db_config = DatabaseConfig()
db_config.connect()

# Create recipe
recipe_crud = RecipeCRUD(db_config.db)
recipe = RecipeModel.create_recipe(
    recipe_id="REC001",
    title="Simple Pasta",
    description="A basic pasta recipe",
    category="Main Course",
    prep_time=10,
    cook_time=20,
    servings=4,
    difficulty="Easy",
    ingredients=[
        RecipeModel.create_ingredient("pasta", 500, "grams"),
        RecipeModel.create_ingredient("sauce", 400, "ml")
    ],
    instructions=[
        RecipeModel.create_instruction(1, "Boil water"),
        RecipeModel.create_instruction(2, "Cook pasta")
    ]
)
result = recipe_crud.create(recipe)
```

### Querying Recipes
```python
# Read by ID
recipe = recipe_crud.read_by_id("REC001")

# Read by category
main_courses = recipe_crud.read_by_category("Main Course")

# Full-text search
results = recipe_crud.search_by_text("chocolate")

# Search by tags
tagged_recipes = recipe_crud.read_by_tags(["vegetarian", "healthy"])
```

### Updating Recipes
```python
# Update fields
recipe_crud.update("REC001", {
    "title": "Updated Recipe Title",
    "description": "New description"
})

# Update rating
recipe_crud.update_rating("REC001", 4.5, review_count=10)

# Add tag
recipe_crud.add_tag("REC001", "quick")
```

### Deleting Recipes
```python
# Delete recipe
recipe_crud.delete("REC001")
```

## Assignment Components

### 1. Database Setup and Configuration (5 marks)
- **File:** `database_setup.py`
- Handles MongoDB connection, collection creation, and index setup
- Includes error handling and connection validation

### 2. Data Model Design (10 marks)
- **File:** `data_model.py`
- Demonstrates flexible schema design
- Includes embedded documents and arrays
- Shows how different recipes can have different structures

### 3. CRUD Operations Implementation (10 marks)
- **File:** `crud_operations.py`
- Complete implementation of all CRUD operations
- Multiple read operations (by ID, category, difficulty, tags, text search)
- Update operations including array operations
- Delete operations

### 4. Report and Demonstration (5 marks)
- **File:** `demonstration.py`
- Complete demonstration of all features
- **File:** `ASSIGNMENT_REPORT.md`
- Detailed report covering all aspects

## Testing

After installation, run the demonstration script to test all functionality:

```bash
python demonstration.py
```

Expected output includes:
- Database connection confirmation
- Collection and index creation
- Sample data creation
- All CRUD operations
- Feature demonstrations

## Troubleshooting

### MongoDB Connection Issues
- Ensure MongoDB service is running
- Check if MongoDB is listening on port 27017
- Verify connection string in `database_setup.py`

### Import Errors
- Ensure all dependencies are installed: `pip install -r requirements.txt`
- Check Python version (3.7+ required)

## Author

**Student:** [Your Name]  
**Course:** BCSE406L - NoSQL Database  
**Date:** November 2025

## License

This project is created for educational purposes as part of the NoSQL Database course assignment.
