#!/usr/bin/env bash
# ./copy_human_images.sh /home/andrey/projects/matting2/data/Adobe/Combined_Dataset

function copy_adobe() {
  while read p; do
    if [ -f "$1/Other/fg/$p" ]; then
      cp $1/Other/fg/$p fg_$2
      cp $1/Other/alpha/"$p" mask_$2
    else
      cp $1/Adobe-licensed\ images/fg/"$p" fg_$2
      cp $1/Adobe-licensed\ images/alpha/"$p" mask_$2
    fi
  done <$2_data_list.txt
}
mkdir -p fg_train fg_test mask_train mask_test merged_train merged_test
copy_adobe "$1/Test_set" test
copy_adobe "$1/Training_set" train
