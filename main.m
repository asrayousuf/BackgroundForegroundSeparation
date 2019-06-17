%% Read images from file and convert to a single matrix
clear all;close all;clc
x_size=576;y_size=720;
siz=x_size*y_size;%enter the dimentions of the image
red_fac=0.5;%compression factor for each image to adjust memory requirements of the system
image_i=100;image_j=500;%enter the image sequence numbers you want to process
A = zeros(siz*red_fac^2,(image_j-image_i)/2);
col=0;
for k = image_i:image_j
    col=col+1;
    img_name=num2str(k);
    if(k/1000<1)
        img_name=strcat('0',img_name);
    end
    filename = strcat('in00',img_name,'.jpg');
    imageFile = fullfile('../PETS2006/input/', filename);
    if exist(imageFile, 'file')
        frame = imread(imageFile);
        frame = imresize(frame,red_fac);
        gray_frame = rgb2gray(frame);
        V = gray_frame(:);
        A(:,col) = V;
    else
        warningMessage = sprintf('Warning: image file does not exist:\n%s', imageFile);
        uiwait(warndlg(warningMessage));
    end
end
[m0,n0]=size(gray_frame);

%% ground truth
% ground truth video extraction
GT = zeros(siz*red_fac^2,(image_j-image_i)/2);
col=0;
for k = image_i:image_j
    col=col+1;
    img_name=num2str(k);
    if(k/1000<1)
        img_name=strcat('0',img_name);
    end
    filename = strcat('gt00',img_name,'.png');
    imageFile = fullfile('../PETS2006/groundtruth/', filename);
    frame = imread(imageFile);
    frame = imresize(frame,red_fac);
    V = frame(:);
    GT(:,col) = V;
end
%% check to see if matrix transformation is correct
% extract frame 246 from matrix and compare to actual frame 246
% frame_number=246;
% frame_mat=reshape(A(:,frame_number-image_i+1),[240*red_fac,320*red_fac]);
% I=mat2gray(frame_mat);
% figure,subplot(2,1,1), imshow(I);
% title('actual image ')
% %actual frame
% filename = sprintf('in000%d.jpg', frame_number);
% imageFile = fullfile('database/baseline/baseline/highway/input/', filename);
% frame = imread(imageFile);
% frame = imresize(frame,red_fac);
% gray_frame = rgb2gray(frame);
% V = gray_frame(:);
% act_frame= reshape(V,[240*red_fac,320*red_fac]);
% subplot (2,1,2),imshow(act_frame);
% title('extracted image frame from column of A')
%% SVD
tic
tol=1e-2;
k=2;%compresion factor
[U,S,V]=svd(A);
A_recovered_SVD= U(:,1:k)*S(1:k,:)*V';
toc
disp('time taken for SVD is')

% visualize performance of background removal
%actual videos from A
figure(1); clf;title('actual vs background and foreground of video from decomposed A recovered by NMF')
colormap( 'Gray' );
for k = image_i:image_i+10
    frame_mat=reshape(A(:,k-image_i+1),[m0,n0]);
    I_or=mat2gray(frame_mat);
    subplot(3,1,1),imagesc(I_or);axis off;axis image
    drawnow; pause(.05);
    %background removed frame from A_recovered
    frame_mat=reshape(A_recovered_SVD(:,k-image_i+1),[x_size*red_fac,y_size*red_fac]);
    I_back=mat2gray(frame_mat);
    subplot(3,1,2),imagesc(I_back);axis off;axis image
    drawnow;pause(.05);
    %foreground extraction
    fore=A-A_recovered_SVD;
    frame_mat=reshape(fore(:,k-image_i+1),[x_size*red_fac,y_size*red_fac]);
    I_fore=imbinarize(mat2gray(frame_mat));
    subplot(3,1,3),imagesc(I_fore);axis off;axis image
    drawnow;pause(.05);
end

%convert to binary video
foreground=A-A_recovered_SVD;

% figure,imshow(imbinarize(mat2gray(reshape(GT(:,1),[m0,n0]))));
% figure,imshow(~imbinarize(mat2gray(reshape(foreground(:,1),[m0,n0]))));
binary_video=~imbinarize(mat2gray(foreground));
[precision_svd, recall_svd, f_measure_svd,accuracy_svd] = output_analysis(imbinarize(mat2gray(GT)) , binary_video);
disp(["Precision of SVD:", precision_svd]);
disp(["Recall of SVD: ", recall_svd]);
disp(["f_measure of SVD: ", f_measure_svd]);
disp(["accuracy of SVD: ", accuracy_svd]);
%% RSVD
tic
k=2;% low rank approximation factor
[U,S,V]=random_SVD(A,k);
A_recovered_rsvd= U*S*V';
toc
disp('time taken for RSVD is')

% visualize performance of background removal
%actual videos from A
figure(1); clf;title('actual vs background and foreground of video from decomposed A recovered by NMF')
colormap( 'Gray' );
for k = image_i:image_i+10
    frame_mat=reshape(A(:,k-image_i+1),[m0,n0]);
    I_or=mat2gray(frame_mat);
    subplot(3,1,1),imagesc(I_or);axis off;axis image
    drawnow; pause(.05);
    %background removed frame from A_recovered
    frame_mat=reshape(A_recovered_rsvd(:,k-image_i+1),[x_size*red_fac,y_size*red_fac]);
    I_back=mat2gray(frame_mat);
    subplot(3,1,2),imagesc(I_back);axis off;axis image
    drawnow;pause(.05);
    %foreground extraction
    fore=A-A_recovered_rsvd;
    frame_mat=reshape(fore(:,k-image_i+1),[x_size*red_fac,y_size*red_fac]);
    I_fore=imbinarize(mat2gray(frame_mat));
    subplot(3,1,3),imagesc(I_fore);axis off;axis image
    drawnow;pause(.05);
end

%convert to binary video
foreground=A-A_recovered_rsvd;
binary_video=~imbinarize(mat2gray(foreground));
% figure,imshow(imbinarize(mat2gray(reshape(GT(:,1),[m0,n0]))));
% figure,imshow(~imbinarize(mat2gray(reshape(foreground(:,1),[m0,n0]))));
% performance check
[precision_rsvd, recall_rsvd, f_measure_rsvd,accuracy_rsvd] = output_analysis(imbinarize(mat2gray(GT)) , binary_video);
disp(["Precision of RSVD:", precision_rsvd]);
disp(["Recall of RSVD: ", recall_rsvd]);
disp(["f_measure of RSVD: ", f_measure_rsvd]);
disp(["accuracy of RSVD: ", accuracy_rsvd]);
%% NMF
tic
k=2;
[W,H]=nnmf(A,k);% to compute decompiosition by NNMF
A_recovered_NMF= W*H;
disp('time taken for NMF is')
toc

% visualize performance of background removal
%actual videos from A
figure(1); clf;title('actual vs background and foreground of video from decomposed A recovered by NMF')
colormap( 'Gray' );
for k = image_i:image_i+200
    frame_mat=reshape(A(:,k-image_i+1),[m0,n0]);
    I_or=mat2gray(frame_mat);
    subplot(3,1,1),imagesc(I_or);axis off;axis image
    drawnow; pause(.05);
    %background removed frame from A_recovered
    frame_mat=reshape(A_recovered_NMF(:,k-image_i+1),[x_size*red_fac,y_size*red_fac]);
    I_back=mat2gray(frame_mat);
    subplot(3,1,2),imagesc(I_back);axis off;axis image
    drawnow;pause(.05);
    %foreground extraction
    fore=A-A_recovered_NMF;
    frame_mat=reshape(fore(:,k-image_i+1),[x_size*red_fac,y_size*red_fac]);
    I_fore=imbinarize(mat2gray(frame_mat));
    subplot(3,1,3),imagesc(I_fore);axis off;axis image
    drawnow;pause(.05);
end
%convert to binary video
foreground=A-A_recovered_NMF;

% figure,imshow(imbinarize(mat2gray(reshape(GT(:,1),[m0,n0]))));
% figure,imshow(~imbinarize(mat2gray(reshape(foreground(:,1),[m0,n0]))));
binary_video=~imbinarize(mat2gray(foreground));
[precision_nmf, recall_nmf, f_measure_nmf,accuracy_nmf] = output_analysis(imbinarize(mat2gray(GT)) , binary_video);
disp(["Precision of NMF:", precision_nmf]);
disp(["Recall of NMF: ", recall_nmf]);
disp(["f_measure of NMF: ", f_measure_nmf]);
disp(["accuracy of NMF: ", accuracy_nmf]);
%% Robust PCA

k=2;%rank compression factor
num_iter=10;
tic
[L, S, res] = pcp(A, num_iter,k);% to compute decomposition by robust PCA
disp('time taken for RPCA is')
toc

% visualization
mat  = @(x) (mat2gray(reshape( x, [x_size*red_fac,y_size*red_fac] )));
nFrames     = size(A,2);
figure(1); clf; 
%title('actual vs background and foreground of video from decomposed A recovered by robust PCA')
colormap( 'Gray' );
for k = 1:200
    %implay(mat(L(:,k)));
    %imagesc( [mat(A(:,k)), mat(L(:,k)),  imbinarize(mat(S(:,k)))] );
    % compare it to just using the median
    %title('Original Image');
     subplot(3,1,1),imagesc(mat(A(:,k)));axis off;axis image
     drawnow;pause(.05);
   %  title('Stationary Background');
     subplot(3,1,2),imagesc( mat(L(:,k)));axis off;axis image
    drawnow;pause(.05);
    %title('Moving Foreground');
    subplot(3,1,3),imagesc(mat(S(:,k)));
     axis off;axis image
    drawnow;
    pause(.05);
    %     imagesc( [mat(X(:,k)), mat(L0(:,k)),  mat(S0(:,k))] );
    
end
binary_video=~imbinarize(mat2gray(S));
[precision_pca, recall_pca, f_measure_pca,accuracy_pca] = output_analysis(imbinarize(mat2gray(GT)) , binary_video);
disp(["Precision of robust PCA:", precision_pca]);
disp(["Recall of robust PCA: ", recall_pca]);
disp(["f_measure of robust PCA: ", f_measure_pca]);
disp(["accuracy of robust PCA: ", accuracy_pca]);

%% plot reduced matrices by all methods
figure, subplot(5,1,1),plot(A);
title('matrix of the original video sequence')
subplot (5,1,2),plot(A_recovered_SVD);
title('matrix of the decomposed (rank deficient) matrix by SVD')
subplot (5,1,3),plot(A_recovered_rsvd);
title('matrix of the decomposed (rank deficient) matrix by RSVD')
subplot (5,1,4),plot(A_recovered_NMF);
title('matrix of the decomposed (rank deficient) matrix by NMF')
subplot (5,1,5),plot(L);
title('matrix of the decomposed (rank deficient) matrix by robust PCA')

%% NMF code with least square minimization
[m,n]= size(A);
% Special case, if k is the rank of original matrix, we know the answer
if k==m
    W = A;
    H = eye(k);
elseif k==n
    W = eye(k);
    H = A;
end


%% NMF code with KL divergence minimization




