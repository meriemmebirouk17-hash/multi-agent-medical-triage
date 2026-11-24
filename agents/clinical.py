{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": []
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "code",
      "source": [
        "import pandas as pd, numpy as np, joblib, sklearn, sys\n",
        "from sklearn.model_selection import train_test_split, cross_val_predict\n",
        "from sklearn.compose import ColumnTransformer\n",
        "from sklearn.pipeline import Pipeline\n",
        "from sklearn.impute import SimpleImputer\n",
        "from sklearn.preprocessing import OneHotEncoder\n",
        "from sklearn.ensemble import RandomForestClassifier\n",
        "from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve\n",
        "\n",
        "df = pd.read_csv(\"healthcare-dataset-stroke-data.csv\")\n",
        "df = df.drop(columns=[\"id\"])\n",
        "df = df[df[\"gender\"] != \"Other\"]\n",
        "\n",
        "X = df.drop(columns=[\"stroke\"])\n",
        "y = df[\"stroke\"]\n",
        "\n",
        "num_cols = [\"age\",\"hypertension\",\"heart_disease\",\"avg_glucose_level\",\"bmi\"]\n",
        "cat_cols = [\"gender\",\"ever_married\",\"work_type\",\"Residence_type\",\"smoking_status\"]\n",
        "\n",
        "prep = ColumnTransformer([\n",
        "    (\"num\", SimpleImputer(strategy=\"median\"), num_cols),\n",
        "    (\"cat\", OneHotEncoder(handle_unknown=\"ignore\"), cat_cols),\n",
        "])\n",
        "\n",
        "pipe = Pipeline([\n",
        "    (\"prep\", prep),\n",
        "    (\"rf\", RandomForestClassifier(n_estimators=300, class_weight=\"balanced_subsample\",\n",
        "                                  min_samples_leaf=5, random_state=42, n_jobs=-1)),\n",
        "])\n",
        "\n",
        "X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)\n",
        "pipe.fit(X_tr, y_tr)\n",
        "\n",
        "proba_cv = cross_val_predict(pipe, X_tr, y_tr, cv=5, method=\"predict_proba\")[:,1]\n",
        "p, r, t = precision_recall_curve(y_tr, proba_cv)\n",
        "f2 = 5*p*r/(4*p+r+1e-9)\n",
        "THRESHOLD = float(t[np.argmax(f2[:-1])])\n",
        "print(\"Seuil retenu :\", round(THRESHOLD,3))\n",
        "\n",
        "proba_te = pipe.predict_proba(X_te)[:,1]\n",
        "pred_te = (proba_te >= THRESHOLD).astype(int)\n",
        "print(classification_report(y_te, pred_te))\n",
        "print(confusion_matrix(y_te, pred_te))\n",
        "\n",
        "joblib.dump({\"pipeline\": pipe, \"threshold\": THRESHOLD}, \"stroke_rf.joblib\")\n",
        "print(\"Fichier sauvegardé !\")"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "LxWkSM0NCel3",
        "outputId": "60456ad6-3010-4646-eaa3-acd367981517"
      },
      "execution_count": 1,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Seuil retenu : 0.298\n",
            "              precision    recall  f1-score   support\n",
            "\n",
            "           0       0.98      0.80      0.88       972\n",
            "           1       0.16      0.74      0.26        50\n",
            "\n",
            "    accuracy                           0.80      1022\n",
            "   macro avg       0.57      0.77      0.57      1022\n",
            "weighted avg       0.94      0.80      0.85      1022\n",
            "\n",
            "[[776 196]\n",
            " [ 13  37]]\n",
            "Fichier sauvegardé !\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "from google.colab import files\n",
        "files.download(\"stroke_rf.joblib\")"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 17
        },
        "id": "fp8wqeOwGVlT",
        "outputId": "16f7e0d8-3320-439f-871b-7c43723248c0"
      },
      "execution_count": 2,
      "outputs": [
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<IPython.core.display.Javascript object>"
            ],
            "application/javascript": [
              "\n",
              "    async function download(id, filename, size) {\n",
              "      if (!google.colab.kernel.accessAllowed) {\n",
              "        return;\n",
              "      }\n",
              "      const div = document.createElement('div');\n",
              "      const label = document.createElement('label');\n",
              "      label.textContent = `Downloading \"${filename}\": `;\n",
              "      div.appendChild(label);\n",
              "      const progress = document.createElement('progress');\n",
              "      progress.max = size;\n",
              "      div.appendChild(progress);\n",
              "      document.body.appendChild(div);\n",
              "\n",
              "      const buffers = [];\n",
              "      let downloaded = 0;\n",
              "\n",
              "      const channel = await google.colab.kernel.comms.open(id);\n",
              "      // Send a message to notify the kernel that we're ready.\n",
              "      channel.send({})\n",
              "\n",
              "      for await (const message of channel.messages) {\n",
              "        // Send a message to notify the kernel that we're ready.\n",
              "        channel.send({})\n",
              "        if (message.buffers) {\n",
              "          for (const buffer of message.buffers) {\n",
              "            buffers.push(buffer);\n",
              "            downloaded += buffer.byteLength;\n",
              "            progress.value = downloaded;\n",
              "          }\n",
              "        }\n",
              "      }\n",
              "      const blob = new Blob(buffers, {type: 'application/binary'});\n",
              "      const a = document.createElement('a');\n",
              "      a.href = window.URL.createObjectURL(blob);\n",
              "      a.download = filename;\n",
              "      div.appendChild(a);\n",
              "      a.click();\n",
              "      div.remove();\n",
              "    }\n",
              "  "
            ]
          },
          "metadata": {}
        },
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<IPython.core.display.Javascript object>"
            ],
            "application/javascript": [
              "download(\"download_962f62d5-b519-44af-ae2f-c9f208bd85a1\", \"stroke_rf.joblib\", 6532328)"
            ]
          },
          "metadata": {}
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "print(\"Python\", sys.version.split()[0])\n",
        "print(\"sklearn\", sklearn.__version__)\n",
        "print(\"pandas\", pd.__version__)\n",
        "print(\"numpy\", np.__version__)\n",
        "print(\"joblib\", joblib.__version__)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "mVadhUjMG6lT",
        "outputId": "bf9cb5d8-d5e7-4f91-abf7-cbbc7c533b96"
      },
      "execution_count": 3,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Python 3.13.18\n",
            "sklearn 1.6.1\n",
            "pandas 2.2.3\n",
            "numpy 2.4.6\n",
            "joblib 1.6.0\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "def predict_stroke(clinical_data: dict):\n",
        "    obj = joblib.load(\"stroke_rf.joblib\")\n",
        "    row = pd.DataFrame([clinical_data])[num_cols + cat_cols]\n",
        "    proba = float(obj[\"pipeline\"].predict_proba(row)[0,1])\n",
        "    return {\"class\": int(proba >= obj[\"threshold\"]), \"probability\": round(proba,4),\n",
        "            \"threshold\": round(obj[\"threshold\"],3)}\n",
        "\n",
        "patients = [\n",
        " {\"gender\":\"Male\",\"age\":67,\"hypertension\":0,\"heart_disease\":1,\"ever_married\":\"Yes\",\n",
        "  \"work_type\":\"Private\",\"Residence_type\":\"Urban\",\"avg_glucose_level\":228.69,\"bmi\":36.6,\"smoking_status\":\"formerly smoked\"},\n",
        " {\"gender\":\"Female\",\"age\":25,\"hypertension\":0,\"heart_disease\":0,\"ever_married\":\"No\",\n",
        "  \"work_type\":\"Private\",\"Residence_type\":\"Rural\",\"avg_glucose_level\":85.0,\"bmi\":None,\"smoking_status\":\"never smoked\"},\n",
        " {\"gender\":\"Female\",\"age\":79,\"hypertension\":1,\"heart_disease\":0,\"ever_married\":\"Yes\",\n",
        "  \"work_type\":\"Self-employed\",\"Residence_type\":\"Rural\",\"avg_glucose_level\":174.12,\"bmi\":24.0,\"smoking_status\":\"never smoked\"},\n",
        "]\n",
        "for pt in patients: print(predict_stroke(pt))"
      ],
      "metadata": {
        "id": "QCSg6G_dHcl9",
        "outputId": "c1adba37-253b-410e-8e00-32848a050937",
        "colab": {
          "base_uri": "https://localhost:8080/"
        }
      },
      "execution_count": 4,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "{'class': 1, 'probability': 0.7759, 'threshold': 0.298}\n",
            "{'class': 0, 'probability': 0.0113, 'threshold': 0.298}\n",
            "{'class': 1, 'probability': 0.7795, 'threshold': 0.298}\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "from sklearn.metrics import roc_auc_score, roc_curve\n",
        "from sklearn.linear_model import LogisticRegression\n",
        "from sklearn.preprocessing import StandardScaler\n",
        "import matplotlib.pyplot as plt\n",
        "\n",
        "# 1. AUC du Random Forest\n",
        "auc_rf = roc_auc_score(y_te, proba_te)\n",
        "print(\"AUC Random Forest :\", round(auc_rf, 3))\n",
        "\n",
        "# 2. Comparaison avec une régression logistique (même découpage)\n",
        "prep_lr = ColumnTransformer([\n",
        "    (\"num\", Pipeline([(\"imp\", SimpleImputer(strategy=\"median\")),\n",
        "                      (\"sc\", StandardScaler())]), num_cols),\n",
        "    (\"cat\", OneHotEncoder(handle_unknown=\"ignore\"), cat_cols),\n",
        "])\n",
        "lr = Pipeline([(\"prep\", prep_lr),\n",
        "               (\"lr\", LogisticRegression(class_weight=\"balanced\", max_iter=1000))])\n",
        "lr.fit(X_tr, y_tr)\n",
        "proba_lr = lr.predict_proba(X_te)[:, 1]\n",
        "auc_lr = roc_auc_score(y_te, proba_lr)\n",
        "print(\"AUC Régression logistique : », round(auc_lr, 3)\")\n",
        "\n",
        "# 3. Courbe ROC\n",
        "fpr, tpr, _ = roc_curve(y_te, proba_te)\n",
        "fpr2, tpr2, _ = roc_curve(y_te, proba_lr)\n",
        "plt.figure(figsize=(5, 5))\n",
        "plt.plot(fpr, tpr, label=f\"Random Forest (AUC = {auc_rf:.3f})\")\n",
        "plt.plot(fpr2, tpr2, label=f\"Rég. logistique (AUC = {auc_lr:.3f})\")\n",
        "plt.plot([0, 1], [0, 1], color=\"black\", linestyle=\"dashed\",label=\"hasard\")\n",
        "plt.xlabel(\"Taux de faux positifs\")\n",
        "plt.ylabel(\"Taux de vrais positifs (rappel)\")\n",
        "plt.legend()\n",
        "plt.tight_layout()\n",
        "plt.savefig(\"roc_stroke.png\", dpi=200)\n",
        "plt.show()\n",
        "\n",
        "# 4. Importance des variables\n",
        "names = pipe.named_steps[\"prep\"].get_feature_names_out()\n",
        "imp = pipe.named_steps[\"rf\"].feature_importances_\n",
        "top = pd.Series(imp, index=names).sort_values(ascending=False).head(8)\n",
        "print(top.round(3))\n",
        "plt.figure(figsize=(6, 4))\n",
        "top[::-1].plot(kind=\"barh\")\n",
        "plt.xlabel(\"Importance\")\n",
        "plt.tight_layout()\n",
        "plt.savefig(\"importance_stroke.png\", dpi=200)\n",
        "plt.show()\n",
        "\n",
        "from google.colab import files\n",
        "files.download(\"roc_stroke.png\")\n",
        "files.download(\"importance_stroke.png\")"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 1000
        },
        "id": "UcagLlzyj00g",
        "outputId": "f4703dc8-cf60-494d-a19d-d09b25633423"
      },
      "execution_count": 8,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "AUC Random Forest : 0.812\n",
            "AUC Régression logistique : », round(auc_lr, 3)\n"
          ]
        },
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 500x500 with 1 Axes>"
            ],
            "image/png": "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\n"
          },
          "metadata": {}
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "num__age                            0.410\n",
            "num__avg_glucose_level              0.166\n",
            "num__bmi                            0.146\n",
            "num__hypertension                   0.034\n",
            "cat__ever_married_Yes               0.030\n",
            "cat__ever_married_No                0.025\n",
            "cat__work_type_children             0.019\n",
            "cat__smoking_status_never smoked    0.018\n",
            "dtype: float64\n"
          ]
        },
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 600x400 with 1 Axes>"
            ],
            "image/png": "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Vlbo27evrPN3794NfX19WFtb48qVK/jwww/Ru3dvhiYiImowDE6vqNzcXLUFzY+r+FRpXd2qZwwuXbpU6bHbs3j48CH+53/+B7/88gsMDAzQq1cvxMTEoEmTJoiJicG7775b5XmWlpa4ePEi7ty5g6CgIOTm5qJ58+bw8PDA6tWrn7uuhvC0rwdQe2NORER1i4/qXlGlpaXP/Kc1rKyspLe015U7d+6ovWPucU2aNIGlpWWdXr++Vff1qI8xrw18VPfq46M6aoxq8qiOwYmIZKvJDxciopdFTX628V11RERERDIxOBERERHJxOBEREREJBODExEREZFMDE5EREREMjE4EREREcnE4EREREQkE4MTERERkUwMTkREREQyMTgRERERycTgRERERCQTgxMRERGRTAxORERERDIxOBERERHJxOBEREREJBODExEREZFMDE5EREREMjE4EREREcnE4EREREQkE4MTERERkUxaDV0AEb18OoUegoZSt6HLoKfICR/W0CUQvZI440REREQkE4MTERERkUwMTkREREQyMTgRERERycTgRPSCc3NzQ2BgYK33GxUVBWNj41rvl4joVcbgRNRIjR07FpcvX27oMoiIXir8OAKiRkqlUkGlUjV0GURELxXOOFGj4+bmhoCAAMybNw8mJiZo1aoVFi1aBADIycmBQqFAWlqa1P727dtQKBRITEwEACQmJkKhUODQoUNwdnaGSqWCu7s7CgoKcODAAdjZ2cHQ0BDjx4/H3bt3a6Xm0tJSzJw5E0ZGRmjevDlCQkIghJCOW1lZYdmyZfD394e+vj4sLS2xd+9e/PHHH/Dy8oK+vj6cnJxw5swZ6Rw+qiMiqjkGJ2qUoqOjoaenh5MnT2LlypVYsmQJjhw5UqM+Fi1ahI0bNyI5ORl5eXnw8fHB2rVrERsbi/j4eBw+fBgbNmyotXq1tLRw6tQprFu3Dv/+97+xZcsWtTZr1qxB7969kZqaimHDhmHChAnw9/fH22+/jXPnzqF9+/bw9/dXC1zVKSkpQVFRkdpGRNSYMThRo+Tk5ITQ0FBYW1vD398f3bp1Q0JCQo36WLZsGXr37g1nZ2dMmTIFx44dw3/+8x84OzujT58+GD16NI4ePVor9Zqbm2PNmjWwtbWFn58fPvjgA6xZs0atzdChQ/Huu+/C2toaCxcuRFFREV5//XWMGTMGNjY2CAoKQkZGBm7cuCH7umFhYTAyMpI2c3PzWrkfIqKXFYMTNUpOTk5qr1u3bo2CgoJn7sPU1BS6urpo166d2r6a9vkkPXr0gEKhkF737NkTWVlZKCsre2I9AODo6FhpX01qCg4ORmFhobTl5eU98z0QEb0KuDicGqUmTZqovVYoFCgvL4eGxqP/lnj8cdbDhw+r7UOhUDyxz/ryz3qetK8mNSmVSiiVylqqkIjo5ccZJ6LHtGjRAgCQn58v7Xt8oXhDOXnypNrrEydOwNraGpqamg1UERFR48QZJ6LHqFQq9OjRA+Hh4Wjbti0KCgqwYMGChi4Lubm5mD17Nt59912cO3cOGzZswOrVqxu6LCKiRofBiegfvvjiC0yZMgUuLi6wtbXFypUrMXDgwAatyd/fH/fu3cMbb7wBTU1NfPjhh5g+fXqD1kRE1BgpRE3em0xEjVpRUdGjd9cFfg0NpW5Dl0NPkRM+rKFLIHppVPxsKywshKGh4VPbco0TERERkUx8VEdUx3Jzc2Fvb1/lsYpPFtfVrXr25tKlS7CwsKiz2oiIqGYYnIjqmJmZ2TO/M8/MzKx2iyEioufCNU5EJFtN1gEQEb0suMaJiIiIqA4wOBERERHJxOBEREREJBODExEREZFMDE5EREREMjE4EREREcnE4EREREQkE4MTERERkUwMTkREREQyMTgRERERycTgRERERCQTgxMRERGRTAxORERERDIxOBERERHJxOBEREREJBODExEREZFMDE5EREREMjE4EREREcnE4EREREQkE4MTERERkUxaDV0AEb18OoUegoZSt876zwkfVmd9ExE9D844EREREcnE4EREREQkE4MTERERkUwMTkREREQyMTjRC83NzQ2BgYENXcYLh+NCRNQw+K46oucwadIk3L59G3v27KnX6+7atQtNmjSp12sSERGDE9EzKSsrg0KhaLDrm5iYNNi1iYgaMz6qa2Tc3NwQEBCAefPmwcTEBK1atcKiRYsAADk5OVAoFEhLS5Pa3759GwqFAomJiQCAxMREKBQKHDp0CM7OzlCpVHB3d0dBQQEOHDgAOzs7GBoaYvz48bh7926t1FxeXl5lvQDwzjvvYPjw4WrtHz58iJYtW2Lr1q3SPc+cORMzZ86EkZERmjdvjpCQEAghpHNKSkowd+5ctGnTBnp6eujevbt0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M2bMkOqysbHBZ599VivXJiJ61SmEEKKhiyCil0NRURGMjIxgHvg1NJS6tdp3TviwWu2PiEiuip9thYWF1b7JijNORERERDIxODUiMTEx0NfXr3Jr27btE485ODi8kDU3ZF11reLT1avaauuRHhER1RzfVdeIvPXWW5U+O6hCkyZNpD/XUtWxhlJdza+qLVu24N69e1UeMzExqedqiIioAoNTI2JgYAADA4OGLqNGXsaaa8M/P/+JiIheDAxORFRjFxYPqrNPqSciepFxjRMRERGRTAxORERERDIxOBERERHJxOBEREREJBODExEREZFMDE5EREREMjE4EREREcnE4EREREQkE4MTERERkUwMTkREREQyMTgRERERycTgRERERCQTgxMRERGRTAxORERERDIxOBERERHJxOBEREREJBODExEREZFMDE5EREREMjE4EREREcmk1dAFENHLp1PoIWgodZ+rj5zwYbVUDRFR/eGMExEREZFMDE5EREREMjE4EREREcnE4EREREQkE4MTUS2bNGkSvL29n6uPxMREKBQK3L59u1ZqIiKi2sHg1EgsWrQIXbp0aegyGoV169YhKiqq3q5XEbIcHBxQVlamdszY2LheayEietUxONFL48GDBw1dgqSqWsrKylBeXg4jIyMYGxvXe02//PILtm3bVu/XJSJqTBicXiLl5eVYuXIlOnToAKVSCQsLCyxfvhwAEBQUBBsbG+jq6qJdu3YICQnBw4cPAQBRUVFYvHgx0tPToVAooFAonnsW4vbt25g6dSpatGgBQ0NDuLu7Iz09HQBw+fJlKBQK/Pzzz2rnrFmzBu3bt5deX7hwAUOGDIG+vj5MTU0xYcIE3Lx5Uzru5uaGmTNnIjAwEM2bN8egQYOqrUuhUGDz5s0YPnw4dHV1YWdnh5SUFFy5cgVubm7Q09NDr169kJ2dLZ2TnZ0NLy8vmJqaQl9fH6+//jq+++47tX6trKywdOlS+Pv7w9DQENOnT0dUVBSMjY2xd+9e2NvbQ6lUIjc3t9KjuvLycoSFhaFt27ZQqVTo3LkzvvnmG7X+9+/fDxsbG6hUKvTr1w85OTnV3us/ffDBBwgNDUVJSckT2+Tm5sLLywv6+vowNDSEj48Pbty4UeNrERE1VgxOL5Hg4GCEh4cjJCQEly5dQmxsLExNTQEABgYGiIqKwqVLl7Bu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Tmpvy586dSq2bNmCyMhIODo6wtXVFVFRUbJmZrS1tXH48GG0bNkSQ4cOhaOjI8LDw2U95pNDQ0MDu3fvxr179/DGG29g6tSpst6NBzzffRERkTwKIYRo6CKIaqp///5wcHDA+vXr1fYvWrQIe/bsQVpaWsMU9oorKiqCkZERzAO/hoZSt0bn5oQPq6OqiIieT8XPtsLCwmrfmMVHdfRSuXXrFhITE5GYmFijD60kIiKqDXxU95KKiYmBvr5+lVvbtm2feMzBwaGhSwcAvPfee0+s8b333nviec7Ozpg0aRI++eQT2Nra1npdK1aseGJdQ4YMqfXrERHRy4WP6l5Sd+7cwY0bN6o81qRJkyf+rbgmTZrA0tKyLkuTpaCg4ImfQm1oaIiWLVvWc0WP/PXXX098x5xKpXruj1N42fFRHRG9iviorhEwMDBQ+5DFl03Lli0bLBw9jYmJSZ38EV4iIno1MDgRUY1dWDyozj7ZnojoRcY1TkREREQyMTgRERERycTgRERERCQTgxMRERGRTAxORERERDIxOBERERHJxOBEREREJBODExEREZFMDE5EREREMjE4EREREcnE4EREREQkE4MTERERkUwMTkREREQyMTgRERERycTgRERERCQTgxMRERGRTAxORERERDIxOBERERHJxOBEREREJJNWQxdARC+fTqGHoKHUld0+J3xYHVZDRFR/OONEREREJBODExEREZFMDE5EREREMjE4EREREcnE4FRDVlZWWLt27ROPu7m5ITAwsN7qocYjKioKxsbGtdpnYmIiFAoFbt++Xav9EhG9quo1OC1atAhdunSpz0vWu127dmHp0qUNXQYAICcnBwqFAmlpaTU+V6FQYM+ePbVeExER0cuMH0dQy0xMTBq6BKpFDx48gLa2dkOXQUREL4gazTiVl5dj5cqV6NChA5RKJSwsLLB8+XLpeFBQEGxsbKCrq4t27dohJCQEDx8+BPDoMcPixYuRnp4OhUIBhUKBqKioZy5cCIFFixbBwsICSqUSZmZmCAgIkI5bWVlh2bJl8Pf3h76+PiwtLbF371788ccf8PLygr6+PpycnHDmzBm1fnfu3AkHBwcolUpYWVlh9erVT61jy5YtMDY2RkJCAoDKj+qsrKywYsUKvPPOOzAwMICFhQU+//xztT6Sk5PRpUsX6OjooFu3btizZ4/s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JTao5x0fIYRwdnYWCxYsqIvyXgjPMkalpaWiV69eYsuWLWLixInCy8urTmvkjBMRAQAePHiAs2fPwsPDQ9qnoaEBDw8PpKSkVHlOSkqKWnsAGDRo0BPbv8yeZXwak9oYn7t37+Lhw4fS30R8lTzv+AghkJCQgMzMTPTt27cuS20wzzpGS5YsQcuWLTFlypT6KJN/q46IHrl58ybKyspgamqqtt/U1BQ///xzledcv369yvbXr1+vszobyrOMT2NSG+MTFBQEMzOzSmH8VfCs41NYWIg2bdqgpKQEmpqa+OyzzzBgwIC6LrdBPMsYHT9+HFu3bkVaWlo9VPgIgxMRETW48PBwxMXFITExETo6Og1dzgvDwMAAaWlpKC4uRkJCAmbPno127drBzc2toUtrcHfu3MGECRMQERGB5s2b19t1GZyICADQvHlzaGpq4saNG2r7b9y48cSFza1atapR+5fZs4xPY/I847Nq1SqEh4fju+++g5OTU12W2WCedXw0NDTQoUMHAECXLl2QkZGBsLCwVzI41XSMsrOzkZOTA09PT2lfeXk5AEBLSwuZmZlo3759rdfJNU5EBADQ1taGi4sLEhISpH3l5eVISEhAz549qzynZ8+eau0B4MiRI09s/zJ7lvFpTJ51fFauXImlS5fi4MGD6NatW32U2iBq6/unvLwcJSUldVFig6vpGHXs2BHnz59HWlqatL311lvo168f0tLSYG5uXjeF1unScyJ6qcTFxQmlUimioqLEpUuXxPTp04WxsbG4fv26EEKICRMmiI8//lhqn5SUJLS0tMSqVatERkaGCA0NFU2aNBHnz59vqFuoUzUdn5KSEpGamipSU1NF69atxdy5c0VqaqrIyspqqFuoUzUdn/DwcKGtrS2++eYbkZ+fL2137txpqFuoUzUdnxUrVojDhw+L7OxscenSJbFq1SqhpaUlIiIiGuoW6lxNx+if6uNddQxORKRmw4YNwsLCQmhra4s33nhDnDhxQjrm6uoqJk6cqNb+66+/FjY2NkJbW1s4ODiI+Pj4eq64ftVkfK5evSoAVNpcXV3rv/B6UpPxsbS0rHJ8QkND67/welKT8Zk/f77o0KGD0NHREU2bNhU9e/YUcXFxDVB1/arpz6DH1UdwUgghRN3MZRERERG9WrjGiYiIiEgmBiciIiIimRiciIiIiGRicCIiIiKSicGJiIiISCYGJyIiIiKZGJyIiIiIZGJwIiIiIpKJwYmIiIhIJgYnIqJGbtKkSfD29m7oMqqUk5MDhUKBtLS0hi6FCACDExERvaAePHjQ0CUQVcLgREREEjc3N3zwwQcIDAxE06ZNYWpqioiICPz999+YPHkyDAwM0KFDBxw4cEA6JzExEQqFAvHx8XBycoKOjg569OiBCxcuqPW9c+dOODg4QKlUwsrKCqtXr1Y7bmVlhaVLl8Lf3x+GhoaYPn062rZtCwBwdnaGQqGAm5sbAOD06dMYMGAAmjdvDiMjI7i6uuLcuXNq/SkUCmzZsgUjRoyArq4urK2tsXfvXrU2Fy9exPDhw2FoaAgDAwP06dMH2dnZ0vEtW7bAzs4OOjo66NixIz777LPnHmN6uTE4ERGRmujoaDRv3hynTp3CBx98gPfffx9jxoxBr169cO7cOQwcOBATJkzA3bt31c776KOPsHr1apw+fRotWrSAp6cnHj58CAA4e/YsfHx8MG7cOJw/fx6LFi1CSEgIoqKi1PpYtWoVOnfujNTUVISEhODUqVMAgO+++w75+fnYtWsXAODOnTuYOHEijh8/jhMnTsDa2hpDhw7FnTt31PpbvHgxfHx88NNPP2Ho0KHw8/PDX3/9BQD47bff0LdvXyiVSnz//fc4e/Ys3nnnHZSWlgIAYmJisHDhQixfvhwZGRlYsWIFQkJCEB0dXetjTi8RQUREjdrEiROFl5eXEEIIV1dX8eabb0rHSktLhZ6enpgwYYK0Lz8/XwAQKSkpQgghjh49KgCIuLg4qc2ff/4pVCqV+Oqrr4QQQowfP14MGDBA7bofffSRsLe3l15bWloKb29vtTZXr14VAERqaupT76GsrEwYGBiIb7/9VtoHQCxYsEB6XVxcLACIAwcOCCGECA4OFm3bthUPHjyoss/27duL2NhYtX1Lly4VPXv2fGot9GrjjBMREalxcnKS/q2pqYlmzZrB0dFR2mdqagoAKCgoUDuvZ8+e0r9NTExga2uLjIwMAEBGRgZ69+6t1r53797IyspCWVmZtK9bt26yarxx4wamTZsGa2trGBkZwdDQEMXFxcjNzX3ivejp6cHQ0FCqOy0tDX369EGTJk0q9f/3338jOzsbU6ZMgb6+vrQtW7ZM7VEeNT5aDV0AERG9WP4ZJBQKhdo+hUIBACgvL6/1a+vp6clqN3HiRPz5559Yt24dLC0toVQq0bNnz0oLyqu6l4q6VSrVE/svLi4GAERERKB79+5qxzQ1NWXVSK8mBiciIqoVJ06cgIWFBQDg1q1buHz5Muzs7AAAdnZ2SEpKUmuflJQEGxubpwYRbW1tAFCblao497PPPsPQoUMBAHl5ebh582aN6nVyckJ0dDQePnxYKWCZmprCzMwMv/zyC/z8/GrUL73aGJyIiKhWLFmyBM2aNYOpqSnmz5+P5s2bS58PNWfOHLz++utYunQpxo4di5SUFGzcuLHad6m1bNkSKpUKBw8exGuvvQYdHR0YGRnB2toa27dvR7du3VBUVISPPvroqTNIVZk5cyY2bNiAcePGITg4GEZGRjhx4gTeeOMN2NraYvHixQgICICRkREGDx6MkpISnDlzBrdu3cLs2bOfdZjoJcc1TkREVCvCw8Px4YcfwsXFBdevX8e3334rzRh17doVX3/9NeLi4tCpUycsXLgQS5YswaRJk57ap5aWFtavX4/NmzfDzMwMXl5eAICtW7fi1q1b6Nq1KyZMmICAgAC0bNmyRvU2a9YM33//PYqLi+Hq6goXFxdERERIs09Tp07Fli1bEBkZCUdHR7i6uiIqKkr6iARqnBRCCNHQRRAR0csrMTER/fr1w61bt2BsbNzQ5RDVKc44EREREcnE4EREREQkEx/VEREREcnEGSciIiIimRiciIiIiGRicCIiIiKSicGJiIiISCYGJyIiIiKZGJyIiIiIZGJwIiIiIpKJwYmIiIhIJgYnIiIiIpn+L1fcDWTy/4b1AAAAAElFTkSuQmCC\n"
          },
          "metadata": {}
        },
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<IPython.core.display.Javascript object>"
            ],
            "application/javascript": [
              "\n",
              "    async function download(id, filename, size) {\n",
              "      if (!google.colab.kernel.accessAllowed) {\n",
              "        return;\n",
              "      }\n",
              "      const div = document.createElement('div');\n",
              "      const label = document.createElement('label');\n",
              "      label.textContent = `Downloading \"${filename}\": `;\n",
              "      div.appendChild(label);\n",
              "      const progress = document.createElement('progress');\n",
              "      progress.max = size;\n",
              "      div.appendChild(progress);\n",
              "      document.body.appendChild(div);\n",
              "\n",
              "      const buffers = [];\n",
              "      let downloaded = 0;\n",
              "\n",
              "      const channel = await google.colab.kernel.comms.open(id);\n",
              "      // Send a message to notify the kernel that we're ready.\n",
              "      channel.send({})\n",
              "\n",
              "      for await (const message of channel.messages) {\n",
              "        // Send a message to notify the kernel that we're ready.\n",
              "        channel.send({})\n",
              "        if (message.buffers) {\n",
              "          for (const buffer of message.buffers) {\n",
              "            buffers.push(buffer);\n",
              "            downloaded += buffer.byteLength;\n",
              "            progress.value = downloaded;\n",
              "          }\n",
              "        }\n",
              "      }\n",
              "      const blob = new Blob(buffers, {type: 'application/binary'});\n",
              "      const a = document.createElement('a');\n",
              "      a.href = window.URL.createObjectURL(blob);\n",
              "      a.download = filename;\n",
              "      div.appendChild(a);\n",
              "      a.click();\n",
              "      div.remove();\n",
              "    }\n",
              "  "
            ]
          },
          "metadata": {}
        },
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<IPython.core.display.Javascript object>"
            ],
            "application/javascript": [
              "download(\"download_890f3e38-4fb2-43a0-b0bc-913f04efa7b6\", \"roc_stroke.png\", 78170)"
            ]
          },
          "metadata": {}
        },
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<IPython.core.display.Javascript object>"
            ],
            "application/javascript": [
              "\n",
              "    async function download(id, filename, size) {\n",
              "      if (!google.colab.kernel.accessAllowed) {\n",
              "        return;\n",
              "      }\n",
              "      const div = document.createElement('div');\n",
              "      const label = document.createElement('label');\n",
              "      label.textContent = `Downloading \"${filename}\": `;\n",
              "      div.appendChild(label);\n",
              "      const progress = document.createElement('progress');\n",
              "      progress.max = size;\n",
              "      div.appendChild(progress);\n",
              "      document.body.appendChild(div);\n",
              "\n",
              "      const buffers = [];\n",
              "      let downloaded = 0;\n",
              "\n",
              "      const channel = await google.colab.kernel.comms.open(id);\n",
              "      // Send a message to notify the kernel that we're ready.\n",
              "      channel.send({})\n",
              "\n",
              "      for await (const message of channel.messages) {\n",
              "        // Send a message to notify the kernel that we're ready.\n",
              "        channel.send({})\n",
              "        if (message.buffers) {\n",
              "          for (const buffer of message.buffers) {\n",
              "            buffers.push(buffer);\n",
              "            downloaded += buffer.byteLength;\n",
              "            progress.value = downloaded;\n",
              "          }\n",
              "        }\n",
              "      }\n",
              "      const blob = new Blob(buffers, {type: 'application/binary'});\n",
              "      const a = document.createElement('a');\n",
              "      a.href = window.URL.createObjectURL(blob);\n",
              "      a.download = filename;\n",
              "      div.appendChild(a);\n",
              "      a.click();\n",
              "      div.remove();\n",
              "    }\n",
              "  "
            ]
          },
          "metadata": {}
        },
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<IPython.core.display.Javascript object>"
            ],
            "application/javascript": [
              "download(\"download_3b57c217-5fc0-485d-9b5b-fadb3830986e\", \"importance_stroke.png\", 53921)"
            ]
          },
          "metadata": {}
        }
      ]
    }
  ]
}
