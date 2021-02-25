#/bin/bash

mkdir -p tmp
mkdir -p eurovoc

cd tmp


echo "Downloading EuroVoc data..."
wget -q -nc --show-progress https://wt-public.emm4u.eu/Resources/Eurovoc/indexing_training/bg-full-eurovoc-1.0.zip
wget -q -nc --show-progress https://wt-public.emm4u.eu/Resources/Eurovoc/indexing_training/cs-full-eurovoc-1.0.zip
wget -q -nc --show-progress https://wt-public.emm4u.eu/Resources/Eurovoc/indexing_training/da-full-eurovoc-1.0.zip
wget -q -nc --show-progress https://wt-public.emm4u.eu/Resources/Eurovoc/indexing_training/de-full-eurovoc-1.0.zip
wget -q -nc --show-progress https://wt-public.emm4u.eu/Resources/Eurovoc/indexing_training/el-full-eurovoc-1.0.zip
wget -q -nc --show-progress https://wt-public.emm4u.eu/Resources/Eurovoc/indexing_training/en-full-eurovoc-1.0.zip
wget -q -nc --show-progress https://wt-public.emm4u.eu/Resources/Eurovoc/indexing_training/es-full-eurovoc-1.0.zip
wget -q -nc --show-progress https://wt-public.emm4u.eu/Resources/Eurovoc/indexing_training/et-full-eurovoc-1.0.zip
wget -q -nc --show-progress https://wt-public.emm4u.eu/Resources/Eurovoc/indexing_training/fi-full-eurovoc-1.0.zip
wget -q -nc --show-progress https://wt-public.emm4u.eu/Resources/Eurovoc/indexing_training/fr-full-eurovoc-1.0.zip
wget -q -nc --show-progress https://wt-public.emm4u.eu/Resources/Eurovoc/indexing_training/hu-full-eurovoc-1.0.zip
wget -q -nc --show-progress https://wt-public.emm4u.eu/Resources/Eurovoc/indexing_training/it-full-eurovoc-1.0.zip
wget -q -nc --show-progress https://wt-public.emm4u.eu/Resources/Eurovoc/indexing_training/lt-full-eurovoc-1.0.zip
wget -q -nc --show-progress https://wt-public.emm4u.eu/Resources/Eurovoc/indexing_training/lv-full-eurovoc-1.0.zip
wget -q -nc --show-progress https://wt-public.emm4u.eu/Resources/Eurovoc/indexing_training/mt-full-eurovoc-1.0.zip
wget -q -nc --show-progress https://wt-public.emm4u.eu/Resources/Eurovoc/indexing_training/nl-full-eurovoc-1.0.zip
wget -q -nc --show-progress https://wt-public.emm4u.eu/Resources/Eurovoc/indexing_training/pl-full-eurovoc-1.0.zip
wget -q -nc --show-progress https://wt-public.emm4u.eu/Resources/Eurovoc/indexing_training/pt-full-eurovoc-1.0.zip
wget -q -nc --show-progress https://wt-public.emm4u.eu/Resources/Eurovoc/indexing_training/ro-full-eurovoc-1.0.zip
wget -q -nc --show-progress https://wt-public.emm4u.eu/Resources/Eurovoc/indexing_training/sk-full-eurovoc-1.0.zip
wget -q -nc --show-progress https://wt-public.emm4u.eu/Resources/Eurovoc/indexing_training/sl-full-eurovoc-1.0.zip
wget -q -nc --show-progress https://wt-public.emm4u.eu/Resources/Eurovoc/indexing_training/sv-full-eurovoc-1.0.zip

echo ""
echo "Extracting data..."
unzip -q \*.zip -d ../eurovoc

cd ../eurovoc

for d in * ; do
    echo "$d/workspace/cf"
	mv "$d/workspace/cf/acquis.cf" "$d"
	mv "$d/workspace/cf/opoce.cf" "$d"
done

find . -type f ! -name '*.cf' -delete
find . -type d -empty -delete

cd ..
rm -r tmp