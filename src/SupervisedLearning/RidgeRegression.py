from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import mglearn


X, y = mglearn.datasets.load_extended_boston()
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 15)

ridge = Ridge().fit(X_train, y_train)

print("Train set score: {:.2f}".format(ridge.score(X_train, y_train)))
print("Test set score: {:.2f}".format(ridge.score(X_test, y_test)))

# test for differen alphas

alphas = [0.1, 0.5, 1, 2, 10]

for a in alphas:
	r = Ridge(alpha=a).fit(X_train, y_train)
	print("***********" + str(a) + "***********")
	print("coef_: {}".format(r.coef_))
	print("Train set score: {:.2f}".format(r.score(X_train, y_train)))
	print("Test set score: {:.2f}".format(r.score(X_test, y_test)))
	
	
# plot _coef for different alphas
ridge = Ridge(alpha=1).fit(X_train, y_train)
ridge10 = Ridge(alpha=10).fit(X_train, y_train)
ridge01 = Ridge(alpha=0.1).fit(X_train, y_train)
lr = LinearRegression().fit(X_train, y_train)

plt.plot(ridge.coef_, 's', label="Ridge alpha=1")
plt.plot(ridge10.coef_, '^', label="Ridge alpha=10")
plt.plot(ridge01.coef_, 'v', label="Ridge alpha=0.1")
plt.plot(lr.coef_, 'o', label="LinearRegression")
plt.xlabel("Coefficient index")
plt.ylabel("Coefficient magnitude")
plt.hlines(0, 0, len(lr.coef_))
plt.ylim(-25, 25)
plt.legend()
plt.show()