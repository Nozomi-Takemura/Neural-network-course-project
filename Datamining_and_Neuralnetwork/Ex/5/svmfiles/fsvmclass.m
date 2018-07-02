function bin=FSVMclass(Xtr,Ytr,type_nummer,epsilon,alpha,b,invoer,par)
% deze funtie bepaald het beeld van een nieuw invoerpatroon
% beeld na training.

%Parameters:
% X=de trainingsvectoren
% Y=bijbehorende labels
% Typenummer= Geeft het type van kernel weer
% epsilon= geeft een criterium op de alpha's. Alpha's kleiner dan epsilon worden weggefilterd.
% alpha  = lijst met supportvectoren (kolom)
% b= is de bias
% invoer = nieuw invoerpatroon
% par er kunnen ook nog extra parameters worden meegegen voor de kernelfunctie.


bin=[];
N=length(invoer(1,:));
while N>1000
    bin=[bin;kernel2(invoer(:,1:1000),Xtr,type_nummer,par)*(alpha.*Ytr)+b*ones(1000,1)];
    invoer=invoer(:,1001:N);
    N=N-1000;
end

bin=[bin;kernel2(invoer,Xtr,type_nummer,par)*(alpha.*Ytr)+b*ones(length(invoer(1,:)),1)];