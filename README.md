# iris_model
iris 데이터 호출 후 decisiontree에 학습시켜줌
```py
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

df_clf=DecisionTreeClassifier(random_state=156)

iris_data=load_iris()
x_train,x_test,y_train,y_test=train_test_split(iris_data.data,iris_data.target,test_size=0.2,random_state=11)
df_clf.fit(x_train,y_train)
pred=df_clf.predict(x_test)
accuracy_score(y_test,pred)
```
# 트리 불러오기
```py
from sklearn.tree import export_graphviz
export_graphviz(df_clf,out_file='iris_tree.dot',max_depth=None,feature_names=iris_data.feature_names,class_names=iris_data.target_names,label='all')
```

```py
# export_graphviz()의 호출 결과로 out_file로 지정된 tree.dot 파일을 생성함. 
#파라미터 조정해보기
export_graphviz(df_clf, out_file="tree.dot", class_names=iris_data.target_names,feature_names = iris_data.feature_names, impurity=True, filled=True)
import graphviz
# 위에서 생성된 tree.dot 파일을 Graphviz 읽어서 Jupyter Notebook상에서 시각화
with open("iris_tree.dot") as f:
    dot_graph = f.read()
graphviz.Source(dot_graph)
```


```py
import seaborn as sns
import numpy as np
%matplotlib inline
dic={}
# feature importance 추출
print("Feature importanes:\n{0}".format(np.round(df_clf.feature_importances_, 3)))

# feature별 importance 매핑
for name, value in zip(iris_data.feature_names, df_clf.feature_importances_):
  dic[name]=value
  print('{0} : {1:.3f}'.format(name, value))
#가로그래프는 내림차순으로 정렬한 후 발표해야함!!

#dictionary{feaure name : feature importance}
# items() 데이터를 생성 후 value 자리의 데이터로 내림차순 정렬

print(dic)
#두번쨰 항목은 원하는 키 기준으로 정렬함.
dic2=sorted(dic.items(),key=lambda x : x[1],reverse=True)
feature=[]
value=[]
for i in dic2:
    feature.append(i[0])
    value.append(i[1])
print(feature,value)

#iris_fi_sorted =sorted(iris_fi.items(),key=lambda x:x[1],reverse=True)
#feature=[feature for feature,value, in iris_fi_sorted]
#value=[value for feature,value, in iris_fi_sorted]
# feature importance를 column 별로 시각화하기
sns.barplot(x=value, y=feature) 
```

```py
from sklearn.datasets import make_classification
import matplotlib.pyplot as plt
%matplotlib inline

plt.title("3 Class values with 2 Features Sample data creation")

# 2차원 시각화를 위해서 피처는 2개, 클래스는 3가지 유형의 분류 샘플 데이터 생성.
# make_classification 데이터를 임의로 만듬
#feature 2 , calss가 3인 데이털르 임의로 만듬,출력 3개(n_class)
X_features, y_labels = make_classification(n_features=2, n_redundant=0, n_informative=2, n_classes=3, n_clusters_per_class=1, random_state=0)

# 그래프 형태로 2개의 피처로 2차원 좌표 시각화, 각 클래스 값은 다른 색깔로 표시됨.
plt.scatter(X_features[:, 0], X_features[:, 1], c=y_labels, s=25, edgecolor='k')
```
```py
import numpy as np

# Classifier의 Decision Boundary를 시각화 하는 함수
def visualize_boundary(model, X, y):
  fig,ax = plt.subplots()
  
  # 학습 데이타 scatter plot으로 나타내기
  ax.scatter(X[:, 0], X[:, 1], c=y, s=25, cmap='rainbow', edgecolor='k',
             clim=(y.min(), y.max()), zorder=3)
  ax.axis('tight')
  ax.axis('off')
  xlim_start , xlim_end = ax.get_xlim()
  ylim_start , ylim_end = ax.get_ylim()
  
  # 호출 파라미터로 들어온 training 데이타로 model 학습 . 
  model.fit(X, y)
  # meshgrid 형태인 모든 좌표값으로 예측 수행. 
  xx, yy = np.meshgrid(np.linspace(xlim_start,xlim_end, num=200),np.linspace(ylim_start,ylim_end, num=200))
  Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
  
  # contourf() 를 이용하여 class boundary 를 visualization 수행. 
  n_classes = len(np.unique(y))
  contours = ax.contourf(xx, yy, Z, alpha=0.3,
                         levels=np.arange(n_classes + 1) - 0.5,
                         cmap='rainbow', clim=(y.min(), y.max()), zorder=1)
 ```
 
 ```py
 from sklearn.tree import DecisionTreeClassifier

# 특정한 트리 생성 제약 없는 결정 트리의 Decision Boundary 시각화.
dt_clf = DecisionTreeClassifier().fit(X_features, y_labels)
visualize_boundary(dt_clf, X_features, y_labels)
#과적합됨
```
```py
for i in range(1,10):
    dt_clf = DecisionTreeClassifier(min_samples_leaf=i).fit(X_features, y_labels)
    visualize_boundary(dt_clf, X_features, y_labels)
```

```py
#2층쯤 괜찮은듯
for i in range(1,10):
    for j in range(1,10):
        dt_clf = DecisionTreeClassifier(max_depth=i,min_samples_leaf=j).fit(X_features, y_labels)
        visualize_boundary(dt_clf, X_features, y_labels)
```
```py
dvct_clf=DecisionTreeClassifier(criterion='entropy').fit(X_features,y_labels)
visualize_boundary(dt_clf,X_features,y_labels)
```

# 앙상블
# bagging (잘안씀,디시즌트리로 예측함,n개로 분리 -> 분리된 데이터를 각 desiciontree가 예측함 => randomforest,병렬 처리됨) 
#->빨리 좋은 성능을 내는
# boostig(adm ,그래디언 부스팅 ,xgboost,lgvm(데이터가 적으면 쓰면안됨,1000만개이상있어야함))
# 직렬로 연결됨,느림, -> nevdia 잘되는애 만들어서 현재 40배 빨라짐

<앙상블>
- 단순/가중 평균(voting)<br>
: linear regression ,k nearest Meighbor , support vector meachine 중 가장괜찮은 모델 사용<br>
- 배깅(bagging)=randomforest<br>
- 부스팅(boosting)<br>
- 스택킹(stacking) : 넷플릭스에서 개발함<br>
- 메타학습(meta-learning)<br>

## Random Forest
- 앙상블 알고리즘
- 비교적 빠른 수행 속도
- 다양한 영역 높은 예측 성능
- 결정 트리 기반

## Hard voting (다수결)
## sotf voting ( 자주 사용)

# voting , FandomForest ,KNN 정확도 비교 해보자




