#!/bin/bash
AUTHORS="Alex Spies"
YEAR="2023"
PROJECT_NAME="pytorch-template"

mv pytorch-template/tests/test_pytorch-template.py pytorch-template/tests/test_$PROJECT_NAME.py
mv pytorch-template/pytorch-template.py pytorch-template/$PROJECT_NAME.py
mv pytorch-template $PROJECT_NAME

# Function to perform safe in-place file editing
safe_sed() {
    local pattern="$1"
    local file="$2"
    local tmp_file=$(mktemp)
    sed "$pattern" "$file" > "$tmp_file" && mv "$tmp_file" "$file"
}

# Process files
find ./ -type f -not -path "./setup_template.sh" -not -path "*/.git/*" | while read file; do
    safe_sed "s/pytorch-template/$PROJECT_NAME/g" "$file"
    safe_sed "s/\[\[AUTHORS\]\]/\"$AUTHORS\"/g" "$file"
    safe_sed "s/\[\[YEAR\]\]/$YEAR/g" "$file"
done
