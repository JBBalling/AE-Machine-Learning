# Gaussian Naive Bayes Model


!["Gaussian-Distribution"](images/1200px-Normal_Distribution_PDF.png)

**Example:** Customer Classification

Optimizing Marketing Campaigns:

Given a group of customers of an e-commerce shop and a limited budget for a marketing campaign. 

Send coupons to specific group of customers, that will likely convert to habitual user (a customer that orders 3 times a week)

Customer data:
  - average bill amoint
  - usage week, month, year

Label:
- 1: user converted
- 0: user not converted


Customer: $\vec{x} = [x_1, x_2, x_3]$
$x_1$ orders twice a week, 7 times in a month and 23.3 times a year


|Habitual|Non Habitual|
|:--------:|:--------:|
|$\vec{x_1}=[2, 7, 23.3]$|$\vec{x_4}=[3, 6, 13.31]$|
|$\vec{x_2}=[1, 3, 49.5]$|$\vec{x_5}=[1, 9, 21.3]$|
|$\vec{x_3}=[0, 1, 16.1]$|$\vec{x_6}=[1, 7, 14.3]$|
|---------|---------|
|$\mu_1=\frac{1+2+0}{3}=1$| $\mu_1 = 1.6$|
|$\sigma_1=\sqrt{\frac{(2-1)²+(1-1)²+(0-1)²}{3}} = 0.81 $| $\sigma_1 = 0.94$


$$P(\vec{x}|K=Habitual)=\frac{1}{\sqrt{2\pi\sigma_K²}}*e^\frac{-(x-\mu_K)²}{2\sigma_K²}$$

|Habitual|Non Habitual|
|:--------:|:--------:|
|$\vec{X_\mu}=[1, 3.66, 29.8]$|$\vec{X_\mu}=[1.6, 7.3, 16.2]$|
|$\vec{X_\sigma}=[0.81, 2.49, 14.1]$|$\vec{X_\sigma}=[0.94, 1.24, 3.59]$|

A customer ordered twice a week, how high is the probability that he will convert to Habitual
$$P(x_1=2|K=Habitual) = \frac{1}{\sqrt{2\pi*0.81²}}*e^\frac{-(2-1)²}{2*0.81²} = 0.23 $$

$$P(K|\vec{x})=\frac{P(K)*\frac{1}{\sqrt{2\pi\sigma_K²}}*e^\frac{-(x-\mu_K)²}{2\sigma_K²}* \frac{1}{\sqrt{2\pi\sigma_K²}}*e^\frac{-(x-\mu_K)²}{2\sigma_K²} * ...}{\sum_C P(K=C)*P(\vec{x}|K=C)}$$


Prior: number of habitual/all_customer


Gaussian Model for numerical and categorical features:


Customer data:
  - average bill amoint
  - usage week, month, year
  - service preference (pickup, takeout or both)

Label:
- 1: user converted
- 0: user not converted


$$P(K|\vec{x})=\frac{P(K)*\frac{1}{\sqrt{2\pi\sigma_K²}}*e^\frac{-(x-\mu_K)²}{2\sigma_K²}* catlikelihood * ...}{\sum_C P(K=C)*P(\vec{x}|K=C)}$$

categorical likelihood: $P(x_i=pickup|K=1)$
-> number of habitual user that pick up / number of habitual user