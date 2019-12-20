

myDir = uigetdir; %gets directory
myFiles = dir(fullfile(myDir,'*.jpg')); %gets all wav files in struct
fileID = fopen('C:\\Users\\singl\\Pictures\\AI project data\\test.csv', 'w');
for k = 1:length(myFiles)
  baseFileName = myFiles(k).name;
  fullFileName = fullfile(myDir,baseFileName);
  % all of your actions for filtering and plotting go here
    I = imread(fullFileName);
    I = rgb2gray(I);
    lbpFeatures = extractLBPFeatures(I,'CellSize',[100 100],'Normalization','None');
    numNeighbors = 8;
    numBins = numNeighbors*(numNeighbors-1)+3;
    lbpCellHists = reshape(lbpFeatures,numBins,[]);
    lbpCellHists = bsxfun(@rdivide,lbpCellHists,sum(lbpCellHists));
    lbpFeatures = reshape(lbpCellHists,1,[]);
    display(fullFileName)

    dlmwrite('C:\\Users\\singl\\Pictures\\AI project data\\test.csv',lbpFeatures,'-append', 'precision', '%.6f', 'newline', 'pc')
end
fclose(fileID);
%dlmwrite('C:\\Users\\singl\\Pictures\\AI project data\\CommaValue.csv',lbpFeatures,'delimiter',',');%
