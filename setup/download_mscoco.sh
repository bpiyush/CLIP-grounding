
echo ":::: | Downloading MSCOCO (2017) validation set (images)"
echo ":::: | NOTE: This takes about 15 minutes on a Mac CPU machine"
echo ""
wget http://images.cocodataset.org/zips/val2017.zip
unzip val2017.zip
rm -rf val2017.zip

echo ":::: | Downloading MSCOCO (2017) validation set (panotic segmentation annotation)"
echo ":::: | NOTE: This takes about 15 minutes on a Mac CPU machine"
echo ""
wget http://images.cocodataset.org/annotations/panoptic_annotations_trainval2017.zip
unzip panoptic_annotations_trainval2017.zip
rm -rf panoptic_annotations_trainval2017.zip

unzip annotations/panoptic_train2017.zip
rm -rf annotations/panoptic_train2017.zip
unzip annotations/panoptic_val2017.zip
rm -rf annotations/panoptic_val2017.zip
mkdir -p annotations/panoptic_segmentation/
mv panoptic_train2017 annotations/panoptic_segmentation/train2017
mv panoptic_val2017 annotations/panoptic_segmentation/val2017

echo ":::: | Downloading Panoptic Narrative Grounding benchmark ..."
echo ""
wget https://lambda004.uniandes.edu.co/panoptic-narrative-grounding/annotations/png_coco_val2017.json

# echo ":::: | Downloading features and panoptic segmentation predictions for val2017 set"
# echo ":::: | NOTE: This is about 10GBs and takes about 30 minutes on a Mac CPU machine"
# wget https://lambda004.uniandes.edu.co/panoptic-narrative-grounding/features/val2017.zip
# unzip val2017.zip
# rm -rf val2017.zip
# cd precomputed_features
# tar -xvzf mask_features.tar.gz
# rm -rf mask_features.tar.gz
# tar -xvzf panoptic_seg_predictions.tar.gz
# rm -rf tar -xvzf panoptic_seg_predictions.tar.gz
# tar -xvzf sem_seg_features.tar.gz
# rm -rf sem_seg_features.tar.gz
# cd ../
# rm -rf val2017.zip

echo "-------------"
echo ":::: | Setting up folder structure"
DATA_DIR=data/panoptic_narrative_grounding
mkdir -p $DATA_DIR

mkdir -p $DATA_DIR/images
mv val2017 $DATA_DIR/images/

# mkdir -p $DATA_DIR/annotations
mv annotations $DATA_DIR/
mv png_coco_val2017.json $DATA_DIR/annotations/

# mkdir -p $DATA_DIR/features/
# mv precomputed_features $DATA_DIR/features/val2017