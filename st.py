# Usage: streamlit run st.py


import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score, confusion_matrix
from sklearn.model_selection import train_test_split
import os
import xlrd
import base64

def dowlondCSV(df, nm):
    csv = df.to_csv(index=False,encoding='utf_8_sig')
    b64 = base64.b64encode(csv.encode(encoding='utf_8_sig')).decode(encoding='utf_8_sig')  # some strings <-> bytes conversions necessary here
    href =f'<a href="data:file/csv; base64,{b64}" download="finaldata.csv">Download csv file</a>'
     
    return href



def putJPG(name):
    from PIL import Image
    image = Image.open(name)

    st.image(image,use_column_width=True)
    
class KNN:
    def __init__(self):
        self.DataProcessed()
        print(self.A1)
        self.A123=pd.concat([self.A1,self.A2,self.A3], ignore_index=True).sort_values(by='Type')

    
    def GetDH24T(self, dfnm, df):
        死亡人數num=[]
        受傷人數num=[]
        HR=[]
        Type=[]
        DaddH=[]
        for row in df.itertuples():
            if dfnm == 'A1' or dfnm == 'A2' :
                str=getattr(row, '死亡受傷人數')
                死亡人數num.append(int(str[2]))
                受傷人數num.append(int(str[6]))
                DaddH.append(int(str[2])+int(str[6]))
                
                if dfnm == 'A1':
                    HR.append(0)
                    Type.append(0)
                else:
                    HR.append(1)
                    Type.append(1)
            else:
                死亡人數num.append(0)
                受傷人數num.append(0)
                DaddH.append(0)
                HR.append(0)
                Type.append(2)

        df['死亡人數num']=死亡人數num
        df['受傷人數num']=受傷人數num
        df['24HRup']=HR
        df['Type']=Type
        df['diedOrhurt']=DaddH
        return df

    def DataProcessed(self):
        init_A1 = pd.read_csv('DATA/NPA_TMA1.csv') 
        init_A2 = pd.read_csv('DATA/NPA_TMA2.csv') 
        init_A3 = pd.read_csv('DATA/NPA_TMA3.csv')

        A1=init_A1.drop([1267,1268])
        A2=init_A2.drop([313255,313256])
        A3=init_A3.drop([153038,153039,153037,0])
        A3.rename(columns={'ACCYMD':'發生時間','PLACE':'發生地點', 'CARTYPE':'車種'}, inplace=True)
        
        self.A1=self.GetDH24T('A1', A1)
        self.A2=self.GetDH24T('A2', A2.sample(n=2000, random_state=123, axis=0))
        self.A3=self.GetDH24T('A3', A3.sample(n=2000, random_state=123, axis=0))


    def KNN(self):
        feature1 = '死亡人數num'; feature2 = '受傷人數num'; feature3='24HRup'; ynm='Type'
        X = self.A123[[feature1,feature2, feature3]];    y = self.A123['Type']
        self.X_train,self.X_test,self.y_train,self.y_test = train_test_split(X,y, test_size=0.3, stratify=y)

        from sklearn.neighbors import KNeighborsClassifier
        self.knn = KNeighborsClassifier()
        self.knn.fit(self.X_train, self.y_train) #model train
        self.ypred = self.knn.predict(self.X_test)

    def confusion_matrix(self):
        from sklearn.metrics import confusion_matrix
        return confusion_matrix(self.y_test, self.ypred, labels=None, sample_weight=None)

    def plot(self):
        feature1 = 'diedOrhurt';   feature2 = '24HRup'
        fig, ax = plt.subplots()

        ax = self.plot_iris(feature1,feature2,np.array(["blue","green","red"]))
        self.plot_iris_class(feature1,feature2,self.knn,ax)   
        plt.show()
        st.pyplot()

    def plot_iris(self, feature1,feature2,cmap):    #-- choose (nameX1,nameX2) in iris.data

        X = self.A123[[feature1,feature2]];    y = self.A123['Type']

        x_min, x_max = X[feature1].min() - .5, X[feature1].max() + .5
        y_min, y_max = X[feature2].min() - .5, X[feature2].max() + .5
        plt.figure(2,figsize=(8,6));   plt.clf()
        p1 = plt.scatter(X[feature1][0:1267], X[feature2][0:1267], c=cmap[0], edgecolor='k')
        p2 = plt.scatter(X[feature1][1267:3267], X[feature2][1267:3267], c=cmap[1], edgecolor='k')
        p3 = plt.scatter(X[feature1][3267:], X[feature2][3267:], c=cmap[2], edgecolor='k')

        plt.xlabel(feature1);          plt.ylabel(feature2)
        plt.xlim(x_min, x_max);        plt.ylim(y_min, y_max)
        plt.legend([p1,p2,p3],['A1','A2','A3'],loc="upper left")
        return(plt)

    def plot_iris_class(self, feature1,feature2,model,plt):  #-- plot_iris + 分類class
        ax = plt.gca()
        xlim = ax.get_xlim();   ylim = ax.get_ylim()

        X = self.A123.loc[:,[feature1, feature2]]
        y = self.A123['Type']
        xx, yy = np.meshgrid(np.linspace(*xlim, num=200), np.linspace(*ylim, num=200))
        model = model.fit(X, self.A123['Type'])   #-- fit the data
        Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
        n_classes = len(np.unique(y))  #-- Create a color plot with the results
        contours = ax.contourf(xx, yy, Z, alpha=0.3, levels=np.arange(n_classes + 1) - 0.5,
                            cmap="rainbow", zorder=1)
        return(plt)

class Linear:
    def __init__(self):
        
        self.df, self.one = self.GetDF()
        self.DataProcessed()


    def GetDF(self):
        data = pd.read_excel('data.xlsx', sheet_name = None) # "data" are all sheets as a dictionary
        xls = pd.ExcelFile('data.xlsx')
        one=data.get('data')
        return pd.DataFrame(one), one
    
    def DataProcessed(self):
        #增加humen_in_traffic
        arr=[]
        for i in self.df['車種']:
            arr.append(i.count(';')+1)
        self.df['humen_in_traffic']=arr

        #增加Moth_and_date&month
        arr=[]
        arr1=[]
        for i in self.df['發生時間']:
            arr.append(str(i[5:10]))
            arr1.append(str(i[5]))
        self.df['Moth_and_date']=arr
        self.df['month']=arr1

        #增加diedOrhurt
        arr=[]
        for row in self.df.itertuples():
            arr.append(getattr(row, '死亡人數num')+getattr(row, '受傷人數num'))
        self.df['diedOrhurt']=arr

        car=[]
        label=[]
        for row in self.df.itertuples():
            inArr=False
            for i in range(0,len(car)):
                
                if getattr(row, '_16') == car[i][0]:
                    inArr=True
                    break
            if inArr==False:
                car.append([getattr(row, '_16'), len(car)])
        for row in self.df.itertuples():
            for i in range(0,len(car)):
                if getattr(row, '_16') == car[i][0]:
                    label.append(car[i][1])
        self.df['CarLebal']=label

    def relationJPG(self):
        import seaborn as sns
        dfData = self.df.corr()
        fig, ax =plt.subplots(figsize=(9, 9))
        ax=sns.heatmap(dfData, annot=True, vmax=1, square=True, cmap="Blues")
        plt.show()
        st.pyplot(fig)
        return plt


    def Linear(self):
        import seaborn as sns
 

        X = self.df[['受傷人數num','humen_in_traffic']];    y = list(self.df['diedOrhurt'])
        indices = np.random.permutation(len(self.one))
        self.Xtrain = self.df[['受傷人數num','humen_in_traffic']].iloc[indices[:-422]]
        self.ytrain =np.array(y)[indices[:-422]]
        self.Xtest = self.df[['受傷人數num','humen_in_traffic']].iloc[indices[-422:]]
        self.ytest = np.array(y)[indices[-422:]]

        from sklearn import linear_model
        self.regr = linear_model.LinearRegression()  
        self.regr = self.regr.fit(self.Xtrain, self.ytrain)  
        #print("regr=", regr.coef_, regr.intercept_)  

        self.ypred = self.regr.predict(self.Xtest)  

    def OuputRPCsv(self):
        np.set_printoptions(precision=2)
        round_ypred=[]
        for i in self.ypred:
            round_ypred.append(round(i, 2))
        return pd.DataFrame({'pred':round_ypred, 'real':self.ytest})

    def MSE(self):
        from sklearn.metrics import mean_squared_error
        return mean_squared_error(self.ytest, self.ypred)

    def plot_linear(self):
        fig, ax = plt.subplots()
        p1=ax.scatter(self.Xtest['humen_in_traffic'],self.ytest,color='blue',marker='x') 
        p2=ax.scatter(self.Xtest['humen_in_traffic'],self.ypred,color='green',marker='+') 
        plt.legend([p1,p2],['real','predict'],loc="upper left")
        plt.xlabel('humen_in_traffic')
        plt.ylabel('diedOrhurt')
        plt.show()

        st.pyplot(fig)
    
    def plot_linearline(self):
        fig, ax = plt.subplots()
        p=ax.scatter(self.df['humen_in_traffic'],self.df['diedOrhurt'],color='blue')
        xlim = ax.get_xlim();              ylim = ax.get_ylim()
        xx = np.linspace(*xlim, num=30);   yy = np.linspace(*ylim, num=30)
        zz = self.regr.intercept_ + self.regr.coef_[0]*xx + self.regr.coef_[1]*yy
        for k in np.arange(len(zz)):  plt.text(xx[k],yy[k],"* {:04.2f}".format(zz[k])) 
        plt.xlabel('humen_in_traffic')
        plt.ylabel('diedOrhurt')
        plt.show()

        st.pyplot(fig)
    
class KMEAN:
    def __init__(self):
        self.DataProcessed()
        print(self.A1)
        self.A123=pd.concat([self.A1,self.A2,self.A3], ignore_index=True).sort_values(by='Type')

    
    def GetDH24T(self, dfnm, df):
        死亡人數num=[]
        受傷人數num=[]
        HR=[]
        Type=[]
        DaddH=[]
        for row in df.itertuples():
            if dfnm == 'A1' or dfnm == 'A2' :
                str=getattr(row, '死亡受傷人數')
                死亡人數num.append(int(str[2]))
                受傷人數num.append(int(str[6]))
                DaddH.append(int(str[2])+int(str[6]))
                
                if dfnm == 'A1':
                    HR.append(0)
                    Type.append(0)
                else:
                    HR.append(1)
                    Type.append(1)
            else:
                死亡人數num.append(0)
                受傷人數num.append(0)
                DaddH.append(0)
                HR.append(0)
                Type.append(2)

        df['死亡人數num']=死亡人數num
        df['受傷人數num']=受傷人數num
        df['24HRup']=HR
        df['Type']=Type
        df['diedOrhurt']=DaddH
        return df

    def DataProcessed(self):
        init_A1 = pd.read_csv('DATA/NPA_TMA1.csv') 
        init_A2 = pd.read_csv('DATA/NPA_TMA2.csv') 
        init_A3 = pd.read_csv('DATA/NPA_TMA3.csv')

        A1=init_A1.drop([1267,1268])
        A2=init_A2.drop([313255,313256])
        A3=init_A3.drop([153038,153039,153037,0])
        A3.rename(columns={'ACCYMD':'發生時間','PLACE':'發生地點', 'CARTYPE':'車種'}, inplace=True)
        
        self.A1=self.GetDH24T('A1', A1)
        self.A2=self.GetDH24T('A2', A2.sample(n=2000, random_state=123, axis=0))
        self.A3=self.GetDH24T('A3', A3.sample(n=2000, random_state=123, axis=0))


    def KMEAN(self):
        feature1 = '死亡人數num';   feature2 = '24HRup'
        X = self.A123[[feature1,feature2]];    y = self.A123['Type']
        self.X_train,self.X_test,self.y_train,self.y_test = train_test_split(X,y, test_size=0.3, stratify=y)

        from sklearn import cluster
        self.kmeans_fit = cluster.KMeans(n_clusters = 3).fit(self.X_train)
        self.ypred = self.kmeans_fit.fit_predict(self.X_test); 

    def confusion_matrix(self):
        from sklearn.metrics import confusion_matrix
        return confusion_matrix(self.y_test, self.ypred, labels=None, sample_weight=None)

    def plot(self):
        feature1 = '死亡人數num';   feature2 = '24HRup'
        fig, ax = plt.subplots()
        ax = self.plot_iris(feature1,feature2,np.array(["blue","green","red"]))
        self.plot_iris_class(feature1,feature2,self.kmeans_fit,ax)   

        plt.show()
        st.pyplot()

    def plot_iris(self, feature1,feature2,cmap):    #-- choose (nameX1,nameX2) in iris.data

        X = self.A123[[feature1,feature2]];    y = self.A123['Type']

        x_min, x_max = X[feature1].min() - .5, X[feature1].max() + .5
        y_min, y_max = X[feature2].min() - .5, X[feature2].max() + .5
        plt.figure(2,figsize=(8,6));   plt.clf()
        p1 = plt.scatter(X[feature1][0:1267], X[feature2][0:1267], c=cmap[0], edgecolor='k')
        p2 = plt.scatter(X[feature1][1267:3267], X[feature2][1267:3267], c=cmap[1], edgecolor='k')
        p3 = plt.scatter(X[feature1][3267:], X[feature2][3267:], c=cmap[2], edgecolor='k')

        plt.xlabel('died_number');          plt.ylabel(feature2)
        plt.xlim(x_min, x_max);        plt.ylim(y_min, y_max)
        plt.legend([p1,p2,p3],['A1','A2','A3'],loc="upper left")
        return(plt)

    def plot_iris_class(self, feature1,feature2,model,plt):  #-- plot_iris + 分類class
        ax = plt.gca()
        xlim = ax.get_xlim();   ylim = ax.get_ylim()

        X = self.A123.loc[:,[feature1, feature2]]
        y = self.A123['Type']
        xx, yy = np.meshgrid(np.linspace(*xlim, num=200), np.linspace(*ylim, num=200))
        model = model.fit(X, self.A123['Type'])   #-- fit the data
        Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
        n_classes = len(np.unique(y))  #-- Create a color plot with the results
        contours = ax.contourf(xx, yy, Z, alpha=0.3, levels=np.arange(n_classes + 1) - 0.5,
                            cmap="rainbow", zorder=1)
        return(plt)

#%%##===== (0B) build up basic libraries =====#####


#%%##===== (1) 主程式與設定參數 (Xfile) =====#####
st.sidebar.title("當場或24小時內死亡之交通事故分析")
st.title("TATD-2401：車禍死亡分析")

#Xfile = "data.xlsx"   

#%%##===== (2) (KDD1,KDD2-3) 車禍數數擷取診斷=====#####
if st.sidebar.checkbox("(1) 車禍數原始數據"):        
    st.header("(1) 車禍數原始數據")
    st.subheader("(1.1)A1(當場或24小時內死亡之交通事故資料)")
    st.dataframe(pd.read_csv('DATA/NPA_TMA1.csv').head(5))
    st.subheader("(1.2)A2(交通事故當事人一人以上受傷(不管事輕傷或重傷)或超過24小時後死亡)")
    st.dataframe(pd.read_csv('DATA/NPA_TMA2.csv').head(5))
    st.subheader("(1.3)A3(交通事故無人受傷，僅有財物(例如車損)上的損失)")
    st.dataframe(pd.read_csv('DATA/NPA_TMA3.csv').head(5))
    #st.write("當場或24小時內死亡之交通事故資料",data.get('data'))
    st.subheader("(1.4)資料數據")
    st.write('* 總共1267筆資料')
 #  st.dataframe(pd.DataFrame([opIndex(X)]))
    st.header("(2) 車禍數衍生數據")
    st.subheader("(2.1) 資料數據處理")   
    putJPG("數據處理.png")
    st.subheader("(2.2) 資料數據說明")   
    st.write('* 1.單一事故對應不只一車種的可能性')
    st.write('* 2.橫向的車種資料最大會到20種人車事故發生')
    st.header("(3) 車禍數探索轉換")   
    st.subheader("(3.1) 資料探索轉換")
    putJPG("資料轉換.png")
    st.write('* 1.車種:主要事故為代表')
    st.write('* 2.發生時間/地點:進行詳細剖析')
    st.write('* 3.發生事故:給予每件事故一個身份證字號')
    st.write('* 4.設立關聯:解決多量車種數據問題')
    st.subheader("(3.2) 資料探索轉換(Excel)")
    putJPG("資料轉換2.png")
    st.write('* 1.總共1267筆資料')
    st.write('* 2.肇事相關人車種共2779筆')
    st.subheader("(3.3) CarLebal")
    intro=pd.DataFrame({'車種':'CarLebal', '重型機車':0,'小貨車':1, '小客車': 2, '曳引車':3, '大貨車':4, '輕型機車':5, '半聯結車':6, '自行車':7, '全聯結車':8, '其他車':9,'大型重型機車':10, '大客車':11, '特種車':12}, index=[''])
    st.table(intro)
    st.write('* 車種表格:主要事故車種的總量')
    st.subheader("(3.4) 各參數欄位說明")
    st.write('* humen_in_traffic: 車種欄位數量')
    st.write('* Moth_and_date:發生時間.1及發生時間.2')
    st.write('* month:發生時間欄位分割')
    st.write('* diedOrhurt:受傷人數+死亡人數')
    st.write('* 發生時段:IF(hr>=19,"晚上",IF(hr>=13,"下午",IF(hr>=7,"早上","凌晨")))')
    st.write('* 事故主要車種(移除行人補足車種):車種欄位第一出現車種')
#%%##===== (3) (KDD3-5) 數據模型  =====#####
if st.sidebar.checkbox("(2) Power BI模型"):
    st.header("(4)Power BI模型")
    st.subheader("(4.1) Power BI車禍統計分析")
    putJPG("Power BI.1.png")
    st.write('* 1.實際2020.01-09月份的A1車禍發生數據分析')
    st.write('* 2.頻繁事故時段:早上')
    st.write('* 3.縣市地區前五大:(高雄,台中,台南,桃園,新北)')
    st.write('* 4.樹狀圖分析高事故車種:機車')
    st.subheader("(4.2) Power BI設立關聯")
    putJPG("Power BI.2.png")
    st.write('* 1.計數以一件事故對應一種車種')
    st.write('* 2.源頭EXCEL資料內增設ITEM NO.')
    st.write('* 3.每個車禍案件,對應發生的各個車種')
    st.write('* 4.POWER BI再設關聯即可有效連動')
    st.write('* 5.未來有效預測,可以加設更多的關聯')    
 #%%##===== (4) (KDD3-5) 車禍模型診斷  =====##### 
#%%##===== (4) (KDD3-5) 車禍模型診斷  =====#####
option = st.sidebar.selectbox( 'AI Model', ('choose AI model', 'KMEAN - 車禍等級分類', 'KNN - 車禍等級分類', '迴歸模型 - 死亡受傷人數預測'))
if option=='KNN - 車禍等級分類' or option=='KMEAN - 車禍等級分類' or option =='迴歸模型 - 死亡受傷人數預測' :
    if option=='KNN - 車禍等級分類':
        st.header("KNN - 車禍等級分類")
        KNN=KNN()
        KNN.KNN()
        st.header("(1) KNN標籤轉換")
        st.subheader("(1.1) Type(種類):依據車禍等級分類")
        intro=pd.DataFrame({'車禍等級':['A1', 'A2', 'A3'], 'Label:Type':[0, 1,2]})
        st.table(intro)
        st.subheader("(1.2) 24HRup(車禍發生超過24小時後有人死亡):依據車禍等級定義")
        intro=pd.DataFrame({'車禍發生超過24小時後有人死亡':['是', '否'], 'Label:24HRup':[1, 0]})
        st.table(intro)
        st.header("(2) KNN混淆矩陣")
        st.write('* 資料總筆數:5267(資料集:A1+A2+A3)')
        st.write('* feature1 = 死亡人數num, feature2 = 受傷人數num, feature3=24HRup')
        st.write(KNN.confusion_matrix())
        st.header("(3) KNN分類圖")
        KNN.plot()
        st.header("(4) 展示KNN Final DataFrame(.CSV)")
        st.dataframe(KNN.A123.head(5))
        if st.sidebar.checkbox("產出KNN Final DataFrame --"):
            st.markdown(dowlondCSV(KNN.A123,  'KNN Final Data'), unsafe_allow_html=True)
    elif option == 'KMEAN - 車禍等級分類':
        st.header("KMEAN - 車禍等級分類")
        KMEAN=KMEAN()
        KMEAN.KMEAN()
        st.header("(1) KMEAN標籤轉換")
        st.subheader("(1.1) 24HRup(車禍發生超過24小時後有人死亡):依據車禍等級定義")
        intro=pd.DataFrame({'車禍發生超過24小時後有人死亡':['是', '否'], 'Label:24HRup':[1, 0]})
        st.table(intro)
        st.header("(2) KMEAN混淆矩陣")
        st.write('* feature1 = 死亡人數num, feature2=24HRup')
        st.write('* 資料筆數:5267(資料集:A1+A2+A3)')
        st.write(KMEAN.confusion_matrix())
        st.header("(3) KMEAN聚類圖")
        KMEAN.plot()
        st.header("(4) 展示KMEAN Final DataFrame(.CSV)")
        st.dataframe(KMEAN.A123.head(5))  
        if st.sidebar.checkbox("產出KMEAN Final DataFrame --"):
            st.markdown(dowlondCSV(KMEAN.A123,  'KMEAN Final Data'), unsafe_allow_html=True)
            #st.write('成功產出KMEAN Final Data.csv')
    elif option == '迴歸模型 - 死亡受傷人數預測':
        st.header("(5.1)死亡受傷人數預測")
        Linear=Linear()
        Linear.Linear()
        st.subheader("(5.1.1) 相關係數圖")
        st.write('* 由相關係數圖決定模型訓練參數:')
        st.write('* Feature1=受傷人數num')
        st.write('* Feature2=humen_in_traffic(車禍總人數)')
        st.write('* y=diedOrhurt(死亡受傷人數)')
        png=Linear.relationJPG()
        st.subheader("(5.1.2) 相關係數圖說明")      
        st.write('* 1.受傷人數和車禍總人數進行模型分析，預測死傷人數')
        st.write('* 2.受傷人數相關係數為0.98，車禍總人數相關係數為0.68')
        st.subheader("(5.1.3) 迴歸方程式")
        st.write('* 資料總筆數:1267筆')
        st.write(Linear.regr.coef_[0], "(受傷人數num)" ) 
        st.write(Linear.regr.coef_[1],"(humen_in_traffic) + ", Linear.regr.intercept_ )   
        Linear.plot_linearline()
        st.subheader("(5.1.4) 展示迴歸模型實際與predict對比(.CSV)")
        dftmp=Linear.OuputRPCsv()
        st.dataframe(dftmp.head(5))
        st.header("(5.2) 迴歸模型評估")
        st.subheader("(5.2.1) 實際與predict對比圖")
        Linear.plot_linear()
        st.subheader("(5.2.2) MSE")
        st.write(Linear.MSE())
        st.subheader("(5.2.3) 展示迴歸模型 Final DataFrame(.CSV)")
        st.dataframe(Linear.df.head(5)) 
        if st.sidebar.checkbox("產出Linear Regression Final Data(.CSV) --"):   
            st.markdown(dowlondCSV(Linear.df,  'Linear Regression Final Data'), unsafe_allow_html=True)
            #st.markdown(dowlondCSV(Linear.df,  'Linear Regression Final Data'), unsafe_allow_html=True)
            #st.write('成功產出Linear Regression Final Data.csv')