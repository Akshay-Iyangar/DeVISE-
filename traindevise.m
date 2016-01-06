load('traindata.mat');
load('classnamevector100.mat');
load('tstdata.mat');
load('ylabelsValidate.mat');
v = traindata(:,1:4096);
tj = classnamevector100;
ylabels = traindata(:,4097);
numImages = 40500;
M = rand(500, 4096);
for k = 1:1
for i = 1:40500
    actuallabel = ylabels(i,:);
    grad = zeros(500,4096);
    cost = 0;
    for j = 1:100
        if j ~= actuallabel
            c = 0.1 - tj(actuallabel, :)*M*v(i,:)' + tj(j, :)*M*v(i,:)';
            if c>0
               grad = grad + (tj(j,:) - tj(actuallabel, :))'*v(i,:);
               cost = cost + c;
            end
        end
    end
    M = M - 0.01*grad;
    if mod(i, 500) == 0
        i, cost
    end
end 
end

valid = tstdata(:,1:4096);
predicted = [];
for i = 1:16362
    c = valid(i,:);
    outputvec = M*c';
    outputvec = outputvec';
    Vec = [];
    for j = 1:100
    X = [outputvec; tj(j,:)];
    Vec = [Vec;pdist(X, 'cosine')]; 
    end
    [minval,ind] = max(Vec(:));
    predicted = [predicted;ind];
end
ylabelsValidate = ylabelsValidate';
a = (ylabelsValidate(1:16362,:) == predicted);
sum(a)*100/16362;