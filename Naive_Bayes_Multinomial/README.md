# Multinomial Naive Bayes Classifier

**Category**: Supervised Learning - Probabilistic Model

**Bayes Theorem**

$$P(spam|\vec{w}) = \frac{P(spam) * P(\vec{w}|spam)}{P(spam) * P(\vec{w}|spam) + P(\neg spam)*P(\vec{w}|\neg spam)}$$


**Examples**: Category-Classification

2 classes:
$$P(spam) * P(\vec{w}|spam) + P(\neg spam)*P(\vec{w}|\neg spam)$$

3 classes:
$$P(tech) *P(\vec{w}|tech) + P(finance) *P(\vec{w}|finance) + P(politics) *P(\vec{w}|politics)$$

$$\sum_{C}{P(K=C)*P(\vec{w}|K=C)}$$

*Condensed Naive Bayes model*
$$P(K|\vec{w}) = \frac{P(K)*P(\vec{w}|K)}{\sum_{C} P(K=C)*P(\vec{w}|K=C)}$$

Denominator doesn't need to be calculated because it always stays the same. Also it will be calculated with:

$$P(K|\vec{w}) = P(K)*P(\vec{w}|K)$$

$$P(K|\vec{w}) = log(P(K))+ log(P(\vec{w}|K))$$

$$P(tag|\vec{w}) = log(P(tag))+log(P(\vec{w}="computer"|tag))+log(P(\vec{w}="hardware"|tag)) ... $$
Prioir probability - same as before relative frequency of class c


Multinomial model is not just using existence of a word and non existence, but the frequency of word in a specific class:

$\vec{w}$ = ["stock", "etf", "fire", "stock", ..., "talk"]
*Likelihoods:*
$$P(stock|K) * P(etf|K) * ... * P(talk|K) * ...$$
$$P(stock|K)² * ... * P(etf|K)¹ * ...$$

$\vec{w}$ = [2, 1, 0, ..., 1]

*Laplace-Soothing*: same as in Naive Binary Classifier

TF-Calculation:

$$TF(travel, article1) = \frac{freq(travel,articel1)}{count(words, article1)}$$

Inverse Document Frequency:

$$IDF(word, D) = log(\frac{count(D)}{count(D,word)})$$

$$TF-IDF score = IDF * TF$$

## Implementing Multinomial Bayes Classification
```
denominator stays the same, so no information gain when calculating
1. calculate priors per class
2. calculate likelihoods per word per class (word_freq_per_tag/total_word_count_per_tag):
for word, tags_map in word_freq_per_tags:
    for tag in tags_map:
        word_likelihoods_per_tag[word][tag] = (word_freq_per_tag[word][tag] + 1) / (total_word_count_per_tag[tag] + 2)
3. calculate posteriors per tag
posteriors_per_tag = {tag: math.log(prior) for tag, prior in self.priors_per_tag.items()}
for word in article:
    for tag in self.tags:
        posteriors_per_tag[tag] = posteriors_per_tag[tag] + math.log(self.likelihoods_per_word_per_tag[word][tag])
return posteriors_per_tag
```
Different Use Cases:

Binary Classification (Spam, not Spam) &rarr; Bernoulli
Categorical Classification (Category of articel) &rarr; Mulitnomial 
Promotions Effect Classification (Habitual Customer, Non Habitual Customer) &rarr; Gaussian
Is a user getting an habitual user after promotion?

