function SVMplotc(X1,Y1,X,Y,alpha,b,grid,type_nummer,par,epsilon)
% This function returns a plot after training the SVM.

% parameters:
% X1 - a matrix of the points that will be used in the plot
% Y1 - labels of these points.
% alpha - lagrange multipliers (or support values)
% b - bias
% X, Y - training points and labels


fprintf('Start Plotting...')


% Bepalen van de grenzen van de plot
xmin=min([X1(1,:) X(1,:)])-5;
xmax=max([X1(1,:) X(1,:)])+5;
ymin=min([X1(2,:) X(2,:)])-5;
ymax=max([X1(2,:) X(2,:)])+5;



num= length(X1(1,:));


Xt=[];

for x1=xmin:(xmax-xmin)/(grid-1):xmax;
   for x2=ymin:(ymax-ymin)/(grid-1):ymax;
      Xt=[Xt [x1;x2]];
   end
end

value=fsvmclass(X,Y,type_nummer,epsilon,alpha,b,Xt,par);
% end
BB=[];
[XX,YY] = meshgrid(xmin:(xmax-xmin)/(grid-1):xmax,ymin:(ymax-ymin)/(grid-1):ymax);
for i=1:grid
    BB=[BB value(1+(i-1)*grid:i*grid,1)];
end
[C,h]=contourf(XX,YY,BB);
colormap cool
colorbar
clabel(C,h,[-1 0 1])
hold on


value=[];
% Plot de invoer


value=sign(fsvmclass(X,Y,type_nummer,epsilon,alpha,b,X1,par));

if isempty(Y1)
    legend('r* positive example','b* negative example')
    for i=1:num
        if value(i)==1
            dot='r*';
        else
            dot='b*';
        end
        plot( X1(1,i), X1(2,i), dot)
        hold on
    end
else
    for i=1:num
        if Y1(i,1)==1
            if value(i)==1
                dot='r*';
            else
                dot='r+';
            end
        else
            if value(i)==-1
                dot='b*';
            else
                dot='b+';
            end
        end
        plot( X1(1,i), X1(2,i), dot)
        hold on
    end
end
    
    hold off
    fprintf('Finished Plotting \n')





