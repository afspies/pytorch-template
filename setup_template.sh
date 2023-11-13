AUTHORS="Alex Spies"
YEAR="2023"
PROJECT_NAME="pytorch-template"

mv pytorch-template/tests/test_pytorch-template.py pytorch-template/tests/test_$PROJECT_NAME.py
mv pytorch-template/pytorch-template.py pytorch-template/$PROJECT_NAME.py
mv pytorch-template $PROJECT_NAME

find ./ -type f -not -path "./setup_template.sh" -not -path "*/.git/*" -exec sed -i -e 's/pytorch-template/'"$PROJECT_NAME"'/g' {} \;
find ./ -type f -not -path "./setup_template.sh" -not -path "*/.git/*" -exec sed -i -e 's/\[\[AUTHORS\]\]/"'"$AUTHORS"'"/g' {} \;
find ./ -type f -not -path "./setup_template.sh" -not -path "*/.git/*" -exec sed -i -e 's/\[\[YEAR\]\]/'"$YEAR"'/g' {} \;