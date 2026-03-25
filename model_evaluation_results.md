# Model Evaluation Results

## Model Comparison Summary

| Model Version | Base Model      | Accuracy | Loss (Val) | Precision (Weighted) | Recall (Weighted) | F1-Score (Weighted) |
|---------------|-----------------|----------|------------|----------------------|-------------------|---------------------|
| **v6_model**  | `yolo11s-cls.pt`| 97.56%   | -          | 98.25%               | 97.56%            | 97.84%              |
| **v5_model**  | `yolov8s-cls.pt`| 96.68%   | -          | 98.13%               | 96.68%            | 97.29%              |
| **v8_model**  | `yolo26n-cls.pt`| 96.78%   | 0.1024     | 97.37%               | 96.78%            | 96.94%              |
| **v7_model**  | `yolo11l-cls.pt`| 96.21%   | -          | 96.65%               | 96.21%            | 96.29%              |

*(Note: Val Loss is currently only recorded explicitly for `v8_model`'s early-stopping metrics)*

***

## Evaluation: `v8_model` (Tested on `industrial_defect_dataset_balanced`)

I evaluated `v8_model` using the same balanced dataset. It achieved a **96.78% overall accuracy**. The early-stopping mechanism saved the optimal weights from Epoch 12.

### Classification Report (`v8_model`)

```text
                 precision    recall  f1-score   support

          crack     1.0000    0.9933    0.9967       600
        crazing     1.0000    1.0000    1.0000        60
   crescent_gap     0.6154    1.0000    0.7619        32
           hole     1.0000    0.9322    0.9649       649
      inclusion     0.7960    1.0000    0.8864       160
         normal     0.9950    0.9900    0.9925       600
        patches     1.0000    1.0000    1.0000        60
 pitted_surface     1.0000    0.9833    0.9916        60
rolled-in_scale     1.0000    1.0000    1.0000        60
           rust     1.0000    1.0000    1.0000       600
        scratch     0.9962    0.9765    0.9863       810
          spots     0.7882    0.6768    0.7283        99
   welding_line     0.5465    0.7231    0.6225        65

       accuracy                         0.9678      3855
      macro avg     0.9029    0.9443    0.9178      3855
   weighted avg     0.9737    0.9678    0.9694      3855
```

### Confusion Matrix (`v8_model`)
![v8 Balanced Confusion Matrix](c:\Projects\HCLTech\Cosmetic Defect Detection Project\defect_project\v8_model\confusion_matrix_normalized.png)

***

## Evaluation: `v6_model` (Tested on `industrial_defect_dataset_balanced`)

I evaluated `v6_model` using the same balanced dataset, and the results show it is actually **slightly better than `v7_model`!** The model achieved an astonishing **97.56% accuracy** across 3,855 validation images.

***

## Evaluation: `v5_model` (Tested on `industrial_defect_dataset_balanced`)

I evaluated `v5_model` using the same balanced dataset. It sits perfectly in the middle between `v6_model` and `v7_model`, achieving a **96.68% overall accuracy**. 

### Classification Report (`v5_model`)

```text
                 precision    recall  f1-score   support

          crack     1.0000    1.0000    1.0000       600
        crazing     1.0000    1.0000    1.0000        60
   crescent_gap     0.7045    0.9688    0.8158        32
           hole     1.0000    0.9245    0.9608       649
      inclusion     0.8444    0.9500    0.8941       160
         normal     0.9967    1.0000    0.9983       600
        patches     1.0000    1.0000    1.0000        60
 pitted_surface     1.0000    0.9667    0.9831        60
  punching_hole     0.0000    0.0000    0.0000         0
rolled-in_scale     1.0000    1.0000    1.0000        60
           rust     1.0000    1.0000    1.0000       600
        scratch     0.9963    0.9963    0.9963       810
          spots     0.8493    0.6263    0.7209        99
   welding_line     0.7255    0.5692    0.6379        65

       accuracy                         0.9668      3855
      macro avg     0.8655    0.8573    0.8577      3855
   weighted avg     0.9813    0.9668    0.9729      3855
```

### Confusion Matrix (`v5_model`)
![v5 Balanced Confusion Matrix](C:\Users\drums\.gemini\antigravity\brain\bc218172-b96b-468e-ba89-1d04f47f3fde\v5_confusion_matrix_balanced.png)

### Classification Report (`v6_model`)

```text
                 precision    recall  f1-score   support

          crack     0.9983    1.0000    0.9992       600
        crazing     1.0000    1.0000    1.0000        60
   crescent_gap     0.7045    0.9688    0.8158        32
           hole     0.9967    0.9245    0.9592       649
      inclusion     0.9398    0.9750    0.9571       160
         normal     0.9983    1.0000    0.9992       600
        patches     0.9677    1.0000    0.9836        60
 pitted_surface     1.0000    0.9833    0.9916        60
  punching_hole     0.0000    0.0000    0.0000         0
rolled-in_scale     1.0000    1.0000    1.0000        60
           rust     1.0000    1.0000    1.0000       600
        scratch     1.0000    0.9951    0.9975       810
          spots     0.8526    0.8182    0.8351        99
   welding_line     0.5714    0.7385    0.6443        65

       accuracy                         0.9756      3855
      macro avg     0.8592    0.8859    0.8702      3855
   weighted avg     0.9825    0.9756    0.9784      3855
```

### Confusion Matrix (`v6_model`)
![v6 Balanced Confusion Matrix](C:\Users\drums\.gemini\antigravity\brain\bc218172-b96b-468e-ba89-1d04f47f3fde\v6_confusion_matrix_balanced.png)

***

## Evaluation: `v7_model` (Tested on `industrial_defect_dataset_balanced`)

I re-ran the evaluation script for `v7_model` using the correct `industrial_defect_dataset_balanced/val` dataset, and the results are **massively improved!** The model achieved an outstanding **96.21% accuracy** across 3,855 validation images.

### Classification Report (`v7_model`)

```text
                 precision    recall  f1-score   support

          crack     0.9950    1.0000    0.9975       600
        crazing     0.9091    1.0000    0.9524        60
   crescent_gap     0.6531    1.0000    0.7901        32
           hole     0.9868    0.9183    0.9513       649
      inclusion     0.8378    0.9688    0.8986       160
         normal     0.9851    0.9950    0.9900       600
        patches     0.9836    1.0000    0.9917        60
 pitted_surface     0.9767    0.7000    0.8155        60
rolled-in_scale     0.8955    1.0000    0.9449        60
           rust     0.9983    0.9917    0.9950       600
        scratch     1.0000    0.9877    0.9938       810
          spots     0.7222    0.6566    0.6878        99
   welding_line     0.5529    0.7231    0.6267        65

       accuracy                         0.9621      3855
      macro avg     0.8843    0.9185    0.8950      3855
   weighted avg     0.9665    0.9621    0.9629      3855
```

### Confusion Matrix (`v7_model`)
![v7 Balanced Confusion Matrix](C:\Users\drums\.gemini\antigravity\brain\bc218172-b96b-468e-ba89-1d04f47f3fde\v7_confusion_matrix_balanced.png)

***

## Evaluation: `v7_model` (Tested on `yolo_formatted_data` - INCORRECT DATASET)

Here are the metrics for each class based on `v7_model`'s predictions over the validation dataset:

```text
                 precision    recall  f1-score   support

          crack     0.0000    0.0000    0.0000         0
         crease     0.0000    0.0000    0.0000        11
   crescent_gap     0.7119    0.9130    0.8000        46
           hole     0.0000    0.0000    0.0000         0
      inclusion     0.5000    0.8558    0.6312       104
         normal     0.0000    0.0000    0.0000         0
       oil_spot     0.0000    0.0000    0.0000        41
  punching_hole     0.0000    0.0000    0.0000        44
rolled-in_scale     0.0000    0.0000    0.0000         0
rolled_in_scale     0.0000    0.0000    0.0000        60
     rolled_pit     0.0000    0.0000    0.0000         7
        scratch     0.0000    0.0000    0.0000         0
      scratches     0.0000    0.0000    0.0000        60
      silk_spot     0.0000    0.0000    0.0000       131
          spots     0.0000    0.0000    0.0000         0
  waist_folding     0.0000    0.0000    0.0000        30
     water_spot     0.0000    0.0000    0.0000        58
   welding_line     0.3622    0.8364    0.5055        55

       accuracy                         0.2736       647
      macro avg     0.0874    0.1447    0.1076       647
   weighted avg     0.1618    0.2736    0.2013       647
```

### Confusion Matrix (`v7_model`)
![v7 Confusion Matrix](C:\Users\drums\.gemini\antigravity\brain\bc218172-b96b-468e-ba89-1d04f47f3fde\v7_confusion_matrix.png)

***

## Evaluation: `v4_model`

I evaluated your model (`v4_model`) on the validation dataset to compute precision, recall, and the F1-score for each class. 

### Classification Report (`v4_model`)

Here are the metrics for each class based on the model's predictions over the validation dataset:

```text
                 precision    recall  f1-score   support

         crease     0.0000    0.0000    0.0000        11
   crescent_gap     0.7368    0.9130    0.8155        46
      inclusion     0.4928    0.9808    0.6559       104
         normal     0.0000    0.0000    0.0000         0
       oil_spot     0.5745    0.6585    0.6136        41
  punching_hole     0.4571    0.7273    0.5614        44
rolled-in_scale     0.0000    0.0000    0.0000         0
rolled_in_scale     0.0000    0.0000    0.0000        60
     rolled_pit     0.0000    0.0000    0.0000         7
        scratch     0.0000    0.0000    0.0000         0
      scratches     0.0000    0.0000    0.0000        60
      silk_spot     0.0000    0.0000    0.0000       131
  waist_folding     0.0000    0.0000    0.0000        30
     water_spot     0.3205    0.4310    0.3676        58
   welding_line     0.6875    0.8000    0.7395        55

       accuracy                         0.4204       647
      macro avg     0.2179    0.3007    0.2502       647
   weighted avg     0.2863    0.4204    0.3363       647
```

*Note: Classes with 0.0 metrics either had no true samples in the validation set (like `normal`, `rolled-in_scale`, `scratch`) or the model did not correctly identify any instances of those defects.*

### Confusion Matrix (`v4_model`)

![Confusion Matrix](C:\Users\drums\.gemini\antigravity\brain\bc218172-b96b-468e-ba89-1d04f47f3fde\confusion_matrix.png)

A confusion matrix has been generated tracking what the model predicted vs what the true label was. You can find it in your project folder here: [c:\Projects\HCLTech\Cosmetic Defect Detection Project\confusion_matrix.png](file:///Projects/HCLTech/Cosmetic%20Defect%20Detection%20Project/confusion_matrix.png)

## Evaluation Script

I also created a reusable script named [evaluate_metrics.py](file:///c:/Projects/HCLTech/Cosmetic%20Defect%20Detection%20Project/evaluate_metrics.py) in your project folder. You can run it any time to get these metrics using:

```bash
python evaluate_metrics.py
```
