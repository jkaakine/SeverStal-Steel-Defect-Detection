# SeverStal: Steel Defect Detection

Purpose is to develope a model that can detect defects in steel. Data and more information is available through [Kaggle](https://www.kaggle.com/c/severstal-steel-defect-detection/overview). 

## Done

1. Explore data 
2. Decode masks
   1. Masks were encoded in RLE. This has been decoded to 256*1600 (same as the provided pictures).
3. Split data
   1. Data has been split into training and test sets. For splitting I used stratified splitting technique where we preserve the same percentage for each target class as in the complete set.

## In progress

1. Segmentation model development
   1. Thinking about using Unet or similar architecture

