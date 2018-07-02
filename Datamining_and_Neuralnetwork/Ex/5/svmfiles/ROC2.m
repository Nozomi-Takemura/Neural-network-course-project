function [AREA,THRES_ROC,TPR_ROC,SPEC_ROC,ACC_ROC,PPV_ROC,NPV_ROC,FPR_ROC,SE,FIG]=ROC2(RESULT,CLASS,PLOT,VAR,FIG)

if nargin < 3 | isempty(PLOT)
   PLOT=0;
   FIG=0;
end

if nargin < 4 | isempty(VAR)
    VAR=' ';
end

FI=find(isfinite(RESULT));
RESULT=(RESULT(FI));
CLASS=CLASS(FI);

NRSAM=size(RESULT,1);
NN=sum(CLASS==0);
NP=sum(CLASS==1);

[RESULT_S,I]=sort(RESULT);
CLASS_S=CLASS(I);

TH=RESULT_S(NRSAM);
SAMNR=NRSAM;
TP=0;
FP=0;
TN=NN;
FN=NP;
TPR=0;
FPR=0;
AREA=0;
Q1B=0;
Q2B=0;
THRES_ROC=[TH];
TPR_ROC=[TPR];
FPR_ROC=[FPR];
SPEC_ROC=[TN/(FP+TN)];
ACC_ROC=[(TP+TN)/(NN+NP)];
PPV_ROC=[NaN];
NPV_ROC=[TN/(TN+FN)];

while ~isempty(TH)
   DELTA=CLASS_S(find(RESULT_S==TH));
   DFP=sum(DELTA==0);
   DTP=sum(DELTA==1);
   TN=TN-DFP;
   AREA=AREA + DFP*TP + 0.5*DFP*DTP;
   Q2B=Q2B+DTP*((TN^2)+(TN*DFP)+((1/3)*(DFP^2)));
   Q1B=Q1B+DFP*((TP^2)+(TP*DTP)+((1/3)*(DTP^2)));
   FP=FP+DFP;
   TP=TP+DTP;
   FN=FN-DTP;
   TPR=TP/(TP+FN);
   FPR=FP/(FP+TN);
   
   SAMNR=max(find(RESULT_S<TH));
   TH=RESULT_S(SAMNR);
   
   TPR_ROC=[TPR_ROC ; TPR];
   FPR_ROC=[FPR_ROC ; FPR];
   THRES_ROC=[THRES_ROC ; TH];
   SPEC_ROC=[SPEC_ROC ; TN/(FP+TN)];
   ACC_ROC=[ACC_ROC ; (TP+TN)/(NN+NP)];
   if (TP+FP)==0
       PPV_ROC=[PPV_ROC ; NaN];
   else
       PPV_ROC=[PPV_ROC ; TP/(TP+FP)];
   end
   if (TN+FN)==0
       NPV_ROC=[NPV_ROC ; NaN];
   else
       NPV_ROC=[NPV_ROC ; TN/(TN+FN)];
   end
end

THRES_ROC=[THRES_ROC ; -1];

AREA=AREA/(NN*NP);
Q2=Q2B/((NN^2)*NP);
Q1=Q1B/(NN*(NP^2));

%Q1=AREA/(2-AREA);
%Q2=2*(AREA^2)/(1+AREA);
SE=sqrt((AREA*(1-AREA) + (NP-1)*(Q1-(AREA^2)) + (NN-1)*(Q2-(AREA^2)))/(NN*NP));

if PLOT==1
   FIG=plot(FPR_ROC,TPR_ROC,'b-');
   title(['ROC (',VAR,')              AREA = ',num2str(AREA),'   SE = ',num2str(SE)])
   xlabel('1 - Specificity')
   ylabel('Sensitivity')
end

if PLOT==2
   figure(FIG)
   hold on
   la=plot(FPR_ROC,TPR_ROC,'r-');
   set(la,'LineWidth',2.5);
   title(['ROC (',VAR,')              AREA = ',num2str(AREA),'   SE = ',num2str(SE)])
   xlabel('1 - Specificity')
   ylabel('Sensitivity')
   input('Press return to continue')
   la=plot(FPR_ROC,TPR_ROC,'w-');
   set(la,'LineWidth',2.5);
   plot(FPR_ROC,TPR_ROC,'b-');
   hold off
end

TPR_ROC=TPR_ROC*100;
FPR_ROC=FPR_ROC*100;
SPEC_ROC=SPEC_ROC*100;
ACC_ROC=ACC_ROC*100;
PPV_ROC=PPV_ROC*100;
NPV_ROC=NPV_ROC*100;
