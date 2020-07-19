library(lubridate)
library(fitdistrplus)
library(dplyr)
library(ggplot2)
library(moments)
library(metRology)
library(car) 
library(leaps)
library(caret)

setwd('D:/NOW/FTEC5580/Project/')
FF = read.csv('F-F_Research_Data_Factors_daily.csv',stringsAsFactors = FALSE)
NVDA = read.csv('NVDA.csv',stringsAsFactors = FALSE)

#set date type
FF$Date = ymd(FF$Date)
NVDA$Date = ymd(NVDA$Date)

#calculate net return
dat =ts(NVDA$Adj.Close)
n = length(dat)
NVDA$Net_return = NA
NVDA$Net_return[2:753] = round((dat[2:n]/dat[1:n-1] - 1)*100,3)
NVDA = NVDA[2:753,]

#merge data
NVDA_FF = merge(NVDA[,c('Date','Net_return')],FF,by="Date",all.x=TRUE)

#fit distritubiton
mytheme = theme(panel.grid.major = element_blank(),
                panel.grid.minor = element_blank(),axis.line = element_line(colour = "black"))


ggplot(NVDA_FF,aes(Net_return))+
  geom_density()+xlab("Net Return")+
  theme_bw() + mytheme

m = mean(NVDA_FF$Net_return)
s = sd(NVDA_FF$Net_return)
sk = skewness(NVDA_FF$Net_return)
kurt = kurtosis(NVDA_FF$Net_return)
print(c(m,s,sk,kurt))

fit_norm = fitdist(NVDA_FF$Net_return,'norm','mle')
fit_logis = fitdist(NVDA_FF$Net_return,'logis','mle')
fit_t=fitdist(NVDA_FF$Net_return,dist="t.scaled",method="mle",
              start=list(df=3.30927158 ,mean=0.22207969,sd=1.80791296))
par(mfrow=c(2,2))

list_model = list(fit_norm, fit_logis,fit_t)
text_model = c("norm", "logis","t")
cdfcomp(list_model, legendtext=text_model)
denscomp(list_model, legendtext=text_model)
qqcomp(list_model, legendtext=text_model)
ppcomp(list_model, legendtext=text_model)
gofstat(list_model, legendtext=text_model)

summary(fit_norm)   # display summary statistics
summary(fit_logis)
summary(fit_t)
colnames(NVDA_FF)
#regression
NVDA_FF$RT_RF = NVDA_FF$Net_return-NVDA_FF$RF
model_ff = lm(RT_RF~Mkt.RF+SMB+HML,data = NVDA_FF)
summary(model_ff)
par(mfrow=c(2,2)) 
plot(model_ff)

qqPlot(model_ff,id.method="identify",simulate=TRUE)
shapiro.test(model_ff$residuals)

ncvTest(model_ff)

durbinWatsonTest(model_ff)

par(mfrow=c(1,3))
crPlots(model_ff)

vif(model_ff)

#Compare CAPM & FF
regsub = regsubsets(RT_RF~Mkt.RF+SMB+HML,data = NVDA_FF,nvmax = 4,nbest = 7) 
regsub_sum = summary(regsub)
names(regsub_sum)
regsub_sum$which

regsub_sum$adjr2[c(1,7)]
regsub_sum$cp[c(1,7)]
regsub_sum$bic[c(1,7)]

train.control = trainControl(method="cv",number=10)   # 10-fold CV
train.control1 = trainControl(method="LOOCV")

reg_cv1 = train(RT_RF~Mkt.RF+SMB+HML,data = NVDA_FF, 
             method="lm", trControl=train.control)
print(reg_cv1)

reg_loocv1 = train(RT_RF~Mkt.RF+SMB+HML,data = NVDA_FF, 
             method="lm", trControl=train.control1)
print(reg_loocv1)


reg_cv2 = train(RT_RF~Mkt.RF,data = NVDA_FF, 
                method="lm", trControl=train.control)
print(reg_cv2)

reg_loocv2 = train(RT_RF~Mkt.RF,data = NVDA_FF, 
                   method="lm", trControl=train.control1)
print(reg_loocv2)

TestMSE1 = as.numeric(reg_cv1$results[2])^2 
TestMSE2 = as.numeric(reg_loocv1$results[2])^2 

TestMSE3 = as.numeric(reg_cv2$results[2])^2 
TestMSE4 = as.numeric(reg_loocv2$results[2])^2 
