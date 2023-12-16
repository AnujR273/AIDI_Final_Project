# Machine Learning Programming Final Project

This project is based on https://github.com/shoaibarham/financialsentimentanalysis

Research Paper: https://arxiv.org/abs/2312.08725

To run the project files:

Download the .ipynb files.

Download the dataset from the link. 

Make sure all the files are stored in correct path.

Open the .ipynb files on google collab or jupyter notebook.

# Code:

# Sentiment Positive

df1 = df[df['Sentiment']=='positive']
words = ' '.join(df1['Sentence'].astype(str))
cleaned_word = ' '.join([word for word in words.split() if not word.startswith('@')])

wordcloud = WordCloud(background_color='black',stopwords=STOPWORDS,
                      width=3000, height=2500).generate(''.join(cleaned_word))
plt.imshow(wordcloud)
plt.axis('off')
plt.show()

# Sentiment Negative

df2 = df[df['Sentiment']=='negative']
words = ' '.join(df2['Sentence'].astype(str))
cleaned_word = ' '.join([word for word in words.split() if not word.startswith('@')])

wordcloud = WordCloud(background_color='white',stopwords=STOPWORDS,
                      width=3000, height=2500).generate(''.join(cleaned_word))
plt.imshow(wordcloud)
plt.axis('off')
plt.show()

# Sentiment Neutral

df3 = df[df['Sentiment']=='neutral']
words = ' '.join(df3['Sentence'].astype(str))
cleaned_word = ' '.join([word for word in words.split() if not word.startswith('@')])

wordcloud = WordCloud(background_color='gray',stopwords=STOPWORDS,
                      width=3000, height=2500).generate(''.join(cleaned_word))
plt.imshow(wordcloud)
plt.axis('off')
plt.show()

# Accuracy:

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

sns.set()

Model = Model[:len(Accuracy)]

results_df = pd.DataFrame({'Models': Model, 'Accuracy': Accuracy})

results_df = results_df.sort_values(by='Accuracy', ascending=False)

plt.figure(figsize=(10, 6))
ax = sns.barplot(x='Models', y='Accuracy', data=results_df, palette='viridis')
ax.set_title('Accuracy Comparison of Different Classification Models')
ax.set_ylabel('Accuracy')
ax.set_xlabel('Classification Models')
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')

for p in ax.patches:
    ax.annotate(f'{p.get_height():.2f}%', (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='center', fontsize=9, color='black', xytext=(0, 5),
                textcoords='offset points')

plt.tight_layout()
plt.show()
