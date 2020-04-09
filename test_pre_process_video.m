clc;clear;

name='sample_video/input/'; %input file location
back_name='sample_video/teaser_back.png'; %captured background image



img_list=dir([name '*_img.png']);

%% bias-gain adjustment

ref=double(imread(back_name))/255;

bias=zeros(length(img_list),3); gain=ones(length(img_list),3);
for i=1:30:length(img_list)
    img=double(imread([name img_list(i).name]))/255;
    mask=double(imread([name strrep(img_list(i).name,'_img','_masksDL')]))/255;
        
        for t=1:10
            mask = imdilate(mask,[0 1 0; 1 1 1; 0 1 0]);
        end
        
        mask1=mask;
        for t=1:300
            mask1 = imdilate(mask1,[0 1 0; 1 1 1; 0 1 0]);
        end
        
        [~,biasR,gainR]=bias_gain_corr(img(:,:,1),ref(:,:,1),mask1-mask);
        [~,biasG,gainG]=bias_gain_corr(img(:,:,2),ref(:,:,2),mask1-mask);
        [~,biasB,gainB]=bias_gain_corr(img(:,:,2),ref(:,:,3),mask1-mask);
             
        bias(i,1)=biasR; bias(i,2)=biasG; bias(i,3)=biasB;
        gain(i,1)=gainR; gain(i,2)=gainG; gain(i,3)=gainB;
end

    B=median(bias,1);
    G=median(gain,1);


    ref_tran(:,:,1)=ref(:,:,1)*G(1)+B(1);
    ref_tran(:,:,2)=ref(:,:,2)*G(2)+B(2);
    ref_tran(:,:,3)=ref(:,:,3)*G(3)+B(3);


%% alignment %%

for i=1:length(img_list)
    img=double(imread([name img_list(i).name]))/255;
    mask=double(imread([name strrep(img_list(i).name,'_img','_masksDL')]))/255;

    [ref_tr,A]=align_tr_projective(img,ref_tran,mask); ref_tr=double(ref_tr)/255;
    
    imwrite(ref_tr,[name strrep(img_list(i).name,'_img','_back')]);

end

%% functions %%


function [cap_tran,biasR,gainR]=bias_gain_corr(orgR,capR,cap_mask)
    cap_mask(cap_mask~=1)=0;

    xR=capR(logical(cap_mask));
    yR=orgR(logical(cap_mask));

    gainR=nanstd(yR)/nanstd(xR);
    biasR=nanmean(yR)-gainR*nanmean(xR);

    cap_tran=capR*gainR+biasR;

end

function [recovered,A] = align_tr_projective(thumb0001_col,thumb0001_back_col,thumb0001_maskDL)

thumb0001_col=uint8(255*thumb0001_col);
thumb0001_back_col=uint8(255*thumb0001_back_col);
thumb0001_maskDL=uint8(255*thumb0001_maskDL);
mask=double(thumb0001_maskDL)==255;

thumb0001=rgb2gray(thumb0001_col); thumb0001_back=rgb2gray(thumb0001_back_col);

ptsOriginal  = detectSURFFeatures(thumb0001);
ptsDistorted = detectSURFFeatures(thumb0001_back);

[featuresOriginal,  validPtsOriginal]  = extractFeatures(thumb0001,  ptsOriginal);
[featuresDistorted, validPtsDistorted] = extractFeatures(thumb0001_back, ptsDistorted);

indexPairs = matchFeatures(featuresOriginal, featuresDistorted);

matchedOriginal  = validPtsOriginal(indexPairs(:,1));
matchedDistorted = validPtsDistorted(indexPairs(:,2));

% figure;
% showMatchedFeatures(thumb0001,thumb0001_back,matchedOriginal,matchedDistorted);
% title('Putatively matched points (including outliers)');

[tform, inlierDistorted, inlierOriginal] = estimateGeometricTransform(...
    matchedDistorted, matchedOriginal, 'projective');


% figure;
% showMatchedFeatures(thumb0001,thumb0001_back,inlierOriginal,inlierDistorted);
% title('Matching points (inliers only)');
% legend('ptsOriginal','ptsDistorted');

outputView = imref2d(size(thumb0001));
recovered  = imwarp(thumb0001_back_col,tform,'OutputView',outputView);

% figure, imagesc(sum(abs(double(thumb0001_col)-double(recovered)),3)/3);

mask=sum(double(recovered),3)==0;
recovered(repmat(mask,[1 1 3]))=thumb0001_col(repmat(mask,[1 1 3]));

recovered(repmat(mask&(double(thumb0001_maskDL)/255),[1 1 3]))=thumb0001_back_col(repmat(mask&(double(thumb0001_maskDL)/255),[1 1 3]));

end

