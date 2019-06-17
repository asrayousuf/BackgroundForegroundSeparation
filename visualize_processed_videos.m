function visualize_processed_videos(A,image_i,image_j,m0,n0,A_recovered,red_fac)
figure(1); clf;
colormap( 'Gray' );
for k = image_i:image_i+20
    frame_mat=reshape(A(:,k-image_i+1),[m0*red_fac,n0*red_fac]);
    I_or=mat2gray(frame_mat);
    subplot(3,1,1),imagesc(I_or);axis off;axis image
    drawnow; pause(.05);
    %background removed frame from A_recovered
    frame_mat=reshape(A_recovered(:,k-image_i+1),[240*red_fac,320*red_fac]);
    I_back=mat2gray(frame_mat);
    subplot(3,1,2),imagesc(I_back);axis off;axis image
    drawnow;pause(.05);
    %foreground extraction
    fore=A-A_recovered;
    frame_mat=reshape(fore(:,k-image_i+1),[240*red_fac,320*red_fac]);
    I_fore=imbinarize(mat2gray(frame_mat));
    subplot(3,1,3),imagesc(I_fore);axis off;axis image
    drawnow;pause(.05);
end