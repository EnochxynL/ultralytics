git init Basketball_Detection
cd Basketball_Detection
git remote add origin https://github.com/tranvietcuong03/Basketball_Detection.git
git fetch
git sparse-checkout set Basketball/
git checkout -b master origin/master
cd ..