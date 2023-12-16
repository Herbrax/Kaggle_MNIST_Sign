# Kaggle Competition 2 - Pierre-Antoine Bernard & Simo Hakim - 20096040 

## Overview
This set of Jupyter notebooks, "CNN-XGBOOST.ipynb" and "RandomForest.ipynb", is designed for the Kaggle competition 2 of the class IFT3395/6390A. These notebooks utilize Convolutional Neural Networks (CNN), XGBoost, and a custom Random Forest Classifier.

We split our work in two files : 

### CNN-XGBOOST.ipynb
In this notebook, we implement CNN and XGBoost models. The process includes data preprocessing, model training, making predictions, and an ensemble method to merge results from different models for a comprehensive final prediction.

### RandomForest.ipynb
This notebook provides a custom implementation of the Random Forest algorithm. It entails preprocessing image data, transforming them into Krawtchouk modes, followed by model training and prediction generation.

## Requirements
- Python 3.x
- Libraries: 
  - `pandas`
  - `numpy`
  - `matplotlib`
  - `tensorflow`
  - `xgboost`
  - `sklearn`

## Data Handling
For both models, the data handling process involves loading the data and preprocessing it. In the CNN and XGBoost models, the image data is reshaped and normalized. For the Random Forest model, image data is transformed into Krawtchouk modes for training and prediction.

## Models
1. **CNN Classifier (CNN-XGBOOST.ipynb):** A TensorFlow and Keras-based CNN model designed for image classification, augmented with image data processing and learning rate scheduling techniques.
2. **XGBoost Classifier (CNN-XGBOOST.ipynb):** This model employs a gradient boosting framework suitable for structured or tabular data, with a focus on hyperparameter tuning.
3. **Custom Random Forest Classifier (RandomForest.ipynb):** It features a unique implementation of the Random Forest algorithm, utilizing Krawtchouk mode transformations of image data.

## Steps and Usage

### CNN-XGBOOST.ipynb
0. **Data Loading and Preprocessing:** Load the dataset, preprocess it by normalizing and reshaping the images suitable for the CNN and XGBoost models.
1. **Model Training:** Initialize and train both the CNN and XGBoost models using the preprocessed data.(Hyperparameter tuning for XGBoost)
2. **Prediction Generation:** Use the trained models to make predictions on the test dataset.
3. **Merging Predictions:** Combine the predictions from both models for final output using a custom merge function.
4. **Saving the Results:** Save the merged predictions to a CSV file for competition submission.
   
   ```python
    # Train XGB Model
    best_xgb_model = startXGBOOST(x_train_xgb, y_train_xgb, x_val_xgb, y_val_xgb)
    # Train CNN Model
    cnn_model = startCNN(features_cnn, labels_cnn, input_shape=(28, 28, 1), num_classes=25)
    # XGBoost Predictions
    xgb_preds_a = best_xgb_model.compute_predictions(normalized_test_a_xgb)
    xgb_preds_b = best_xgb_model.compute_predictions(normalized_test_b_xgb)
    xgb_merged_predictions = merge_predictions(xgb_preds_a, xgb_preds_b)
    # CNN Predictions
    cnn_preds_a = cnn_model.predict(normalized_test_a_cnn)
    cnn_preds_b = cnn_model.predict(normalized_test_b_cnn)
    cnn_merged_predictions = merge_predictions(np.argmax(cnn_preds_a, axis=1), np.argmax(cnn_preds_b, axis=1))
    # Save XGB
    save_predictions_to_csv("xgb_predictions.csv", xgb_merged_predictions)
    # Save CNN
    save_predictions_to_csv("cnn_predictions.csv", cnn_merged_predictions)
    ```

### RandomForest.ipynb
0. **Data Loading and Preprocessing:** Load the training and test data, preprocess them and apply Kwartchouk transformation.
1. **Model Training:** Train the custom Random Forest model using these transformed images.
2. **Hyperparameter Tuning:** Perform grid search for hyperparameter optimization to enhance model performance.
3. **Prediction Generation:** Use the trained models to make predictions on the test dataset.
4. **Saving the Results:** Sum ASCII Values and save the merged predictions to a CSV file for competition submission.

   
    ```python
    # Train the custom Random Forest model
    model = randomForest(nbarbres, nbfeatures, nbexamples, profondeurs)
    model.fit(X_train, y_train)

    # Make predictions on the test data
    predictions_test1 = model.PredictionsFinals(image1_test_kraw, ...)
    predictions_test2 = model.PredictionsFinals(image2_test_kraw, ...)

    # Sum ASCII values to get final predictions
    predictions_char = SumASCII(predictions_test1, predictions_test2)

    # Save predictions to CSV for Kaggle submission
    test_pred = np.c_[np.array(range(len(predictions_char))), np.array(predictions_char)]
    df = pd.DataFrame(test_pred, columns=['id', 'Label'])
    df.to_csv("pred_rf.csv", index=False)
    ```

----------------------------

# README pour la Compétition Kaggle : Techniques Avancées d'Apprentissage Automatique

## Aperçu
Cet ensemble de notebooks, "CNN-XGBOOST.ipynb" et "RandomForest.ipynb", estsont conçus pour la compétition Kaggle 2 du cours IFT3395/6390A. Ces notebooks utilisent des CNNs, XGBoost, et un classificateur Random Forest que l'on à implémenté.

Nous avons réparti notre travail en deux fichiers :

### CNN-XGBOOST.ipynb
Dans ce notebook, nous mettons en œuvre les modèles CNN et XGBoost. Le processus comprend le prétraitement des données, l'entraînement des modèles, la génération de prédictions, et une méthode pour fusionner les résultats ASCII de différents modèles pour obtenir une prédiction finale complète.

### RandomForest.ipynb
Ce notebook fournit une implémentation custom de l'algorithme Random Forest. Il implique le prétraitement des données d'image, leur transformation en modes Krawtchouk, suivi par l'entraînement du modèle et la génération de prédictions.

## Prérequis
- Python 3.x
- Libraries : 
  - `pandas`
  - `numpy`
  - `matplotlib`
  - `tensorflow`
  - `xgboost`
  - `sklearn`

## Traitement des Données
Pour les deux modèles, le processus de traitement des données implique le chargement et le prétraitement des données. Dans les modèles CNN et XGBoost, les données d'image sont remodelées et normalisées. Pour le modèle Random Forest, les données d'image sont transformées en modes Krawtchouk pour l'entraînement et la prédiction.

## Modèles
1. **Classificateur CNN (CNN-XGBOOST.ipynb) :** Un modèle CNN basé sur TensorFlow et Keras conçu pour la classification d'images, augmenté avec des techniques de traitement des données d'image et de planification du taux d'apprentissage.
2. **Classificateur XGBoost (CNN-XGBOOST.ipynb) :** Ce modèle utilise un cadre de boosting par gradient adapté aux données structurées ou tabulaires, avec un accent sur le réglage des hyperparamètres.
3. **Classificateur Random Forest Personnalisé (RandomForest.ipynb) :** Il présente une mise en œuvre unique de l'algorithme Random Forest, utilisant des transformations de mode Krawtchouk des données d'image.

## Étapes et Utilisation

### CNN-XGBOOST.ipynb
0. **Chargement et Prétraitement des Données :** Chargez le jeu de données, prétraitez-le en normalisant et remodelant les images pour qu'elles soient adaptées aux modèles CNN et XGBoost.
1. **Entraînement des Modèles :** Initialisez et entraînez les modèles CNN et XGBoost en utilisant les données prétraitées. (Réglage des hyperparamètres pour XGBoost)
2. **Génération de Prédictions :** Utilisez les modèles entraînés pour faire des prédictions sur l'ensemble de test.
3. **Fusion des Prédictions :** Combinez les prédictions des deux modèles pour une sortie finale en utilisant une fonction de fusion personnalisée.
4. **Sauvegarde des Résultats :** Enregistrez les prédictions fusionnées dans un fichier CSV pour la soumission à la compétition.

   ```python
    # Train XGB Model
    best_xgb_model = startXGBOOST(x_train_xgb, y_train_xgb, x_val_xgb, y_val_xgb)
    # Train CNN Model
    cnn_model = startCNN(features_cnn, labels_cnn, input_shape=(28, 28, 1), num_classes=25)
    # XGBoost Predictions
    xgb_preds_a = best_xgb_model.compute_predictions(normalized_test_a_xgb)
    xgb_preds_b = best_xgb_model.compute_predictions(normalized_test_b_xgb)
    xgb_merged_predictions = merge_predictions(xgb_preds_a, xgb_preds_b)
    # CNN Predictions
    cnn_preds_a = cnn_model.predict(normalized_test_a_cnn)
    cnn_preds_b = cnn_model.predict(normalized_test_b_cnn)
    cnn_merged_predictions = merge_predictions(np.argmax(cnn_preds_a, axis=1), np.argmax(cnn_preds_b, axis=1))
    # Save XGB
    save_predictions_to_csv("xgb_predictions.csv", xgb_merged_predictions)
    # Save CNN
    save_predictions_to_csv("cnn_predictions.csv", cnn_merged_predictions)
    ```

### RandomForest.ipynb
0. **Chargement et Prétraitement des Données :** Chargez les données d'entraînement et de test, prétraitez-les et appliquez la transformation Kwartchouk.
1. **Entraînement du Modèle :** Entraînez le modèle Random Forest personnalisé en utilisant ces images transformées.
2. **Réglage des Hyperparamètres :** Effectuez une recherche par grille pour optimiser les hyperparamètres et améliorer les performances du modèle.
3. **Génération de Prédictions :** Utilisez les modèles entraînés pour faire des prédictions sur l'ensemble de test.
4. **Sauvegarde des Résultats :** Sommez les valeurs ASCII et enregistrez les prédictions fusionnées dans un fichier CSV pour la soumission à la compétition.

    ```python
    # Train the custom Random Forest model
    model = randomForest(nbarbres, nbfeatures, nbexamples, profondeurs)
    model.fit(X_train, y_train)

    # Make predictions on the test data
    predictions_test1 = model.PredictionsFinals(image1_test_kraw, ...)
    predictions_test2 = model.PredictionsFinals(image2_test_kraw, ...)

    # Sum ASCII values to get final predictions
    predictions_char = SumASCII(predictions_test1, predictions_test2)

    # Save predictions to CSV for Kaggle submission
    test_pred = np.c_[np.array(range(len(predictions_char))), np.array(predictions_char)]
    df = pd.DataFrame(test_pred, columns=['id', 'Label'])
    df.to_csv("pred_rf.csv", index=False)
    ```
