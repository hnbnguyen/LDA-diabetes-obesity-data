function a4_20188794
% Function for CISC271, Winter 2021, Assignment #4

    % Read the test data from a CSV file
    dmrisk = csvread('dmrisk.csv',1,0);

    % Columns for the data and labels; DM is diabetes, OB is obesity
    jDM = 2;
    jOB = 16;

    % Extract the data matrices and labels
    XDM = dmrisk(:, (1:size(dmrisk,2))~=jDM);
    yDM = dmrisk(:,jDM);
    XOB = dmrisk(:, (1:size(dmrisk,2))~=jOB);
    yOB = dmrisk(:,jOB);

    % Reduce the dimensionality to 2D using PCA
    [~,rDM] = pca(zscore(XDM), 'NumComponents', 2);
    [~,rOB] = pca(zscore(XOB), 'NumComponents', 2);

    % Find the LDA vectors and scores for each data set
    [qDM zDM qOB zOB] = a4q1(rDM, yDM, rOB, yOB);
    

    % %
    % % PLOT RELEVANT DATA
    % %
    
    % Figure for PCA on Diabetes data
    color = lines(6);
    f1 = figure;
    gscatter(rDM(:, 1),rDM(:, 2), yDM, color(1:3,:));
    title("PCA on Diabetes Data")

    % Figure for PCA on Obesity data
    f2 = figure;
    gscatter(rOB(:, 1),rOB(:, 2), yOB, color(1:3,:));
    title("PCA on Obesity Data")
    
    %Figure for LDA score on Diabetes data
    f3 = figure;
    gscatter(zDM, zDM, yDM, 'kg');
    title('LDA scores for Diabetes')
    xlabel('zDM')
    ylabel('zDM')
    
    %Figure for LDA score on Obesity data
    f4 = figure;
    gscatter(zOB, zOB, yOB, 'rb');
    title('LDA scores for Obesity')
    xlabel('zOB')
    ylabel('zOB')
    
    
    % Compute the ROC curve and its AUC where: "xroc" is the horizontal
    % axis of false positive rates; "yroc" is the vertical
    % axis of true positive rates; "auc" is the area under curve
    % %
    % % COMPUTE, PLOT, DISPLAY RELEVANT DATA
    % %
    
    [xrocOB, yrocOB, aucOB, botpOB] = roccurve(yOB, zOB);
    [xrocDM, yrocDM, aucDM, botpDM] = roccurve(yDM, zDM);
    
    % displaying threshold botp and auc for Obesity and Diabetes data
    disp("Best Threshold value for Obesity Data")
    disp(botpOB)
    disp("Best Threshold value for Diabetes Data")
    disp(botpDM)
    disp("AUC for Obesity Data")
    disp(aucOB)
    disp("AUC for Diabetes Data")
    disp(aucDM)
    
    % Figure for ROC curve on Obesity data
    figure5 = figure;
    plot(xrocOB, yrocOB, '.b')
    title('ROC Curve for Obesity')
    xlabel('False Positive Rate')
    ylabel('True Positive Rate')
    
    % Figure for ROC curve on Diabetes data
    figure6 = figure;
    plot(xrocDM, yrocDM, '.r')
    title('ROC Curve for Diabetes')
    xlabel('False Positive Rate')
    ylabel('True Positive Rate')
    
    % getting the confusion matrix with the best threshold
    % for Obesity and Diabetes data and displaying them
    cmOB = confmat(yOB, zOB, botpOB);
    cmDM = confmat(yDM, zDM, botpDM);
   
    disp("Confusion matrix for Obesity Data")
    disp(cmOB)
    disp("Confusion matrix for Diabetes Data")
    disp(cmDM)
    
    A = [ 0,  4,  1; 3,  5,  1; 3,  3,  1; 1,  3,  1; 3,  7,  0; 4,  4,  0; -1,  2,  0; 0,  8,  0];
    Amat = A(:, 1:(end-1));
    yvechw10 = A(:, end);
   [qA, zA, qB, zB] = a4q1(Amat, yvechw10, Amat, yvechw10);
   qA;
   qB;
   sort(zA);
   sort(zB);
   [xrocOB, yrocOB, aucOB, botpOB] = roccurve(yvechw10, zA);
   
   

% END OF FUNCTION
end

function [q1, z1, q2, z2] = a4q1(Xmat1, yvec1, Xmat2, yvec2)
% [Q1 Z1 Q2 Z2]=A4Q1(X1,Y1,X2,Y2) computes an LDA axis and a
% score vector for X1 with Y1, and for X2 with Y2.
%
% INPUTS:
%         X1 - MxN data, M observations of N variables
%         Y1 - Mx1 labels, +/- computed as ==/~= 1
%         X2 - MxN data, M observations of N variables
%         Y2 - Mx1 labels, +/- computed as ==/~= 1
% OUTPUTS:
%         Q1 - Nx1 vector, LDA axis of data set #1
%         Z1 - Mx1 vector, scores of data set #1
%         Q2 - Nx1 vector, LDA axis of data set #2
%         Z2 - Mx1 vector, scores of data set #2

    q1 = [];
    z1 = [];
    q2 = [];
    z2 = [];
    
    % Compute the LDA axis for each data set
    q1 = lda2class(Xmat1(yvec1==1,:), Xmat1(yvec1~=1, :));
    q2 = lda2class(Xmat2(yvec2==1,:), Xmat2(yvec2~=1, :));
   
    % %
    % % COMPUTE SCORES USING LDA AXES
    % %
    z1 = Xmat1 * q1;
    z2 = Xmat2 * q2;
    
    
% END OF FUNCTION
end

function qvec = lda2class(X1, X2)
% QVEC=LDA2(X1,X2) finds Fisher's linear discriminant axis QVEC
% for data in X1 and X2.  The data are assumed to be sufficiently
% independent that the within-label scatter matrix is full rank.
%
% INPUTS:
%         X1   - M1xN data with M1 observations of N variables
%         X2   - M2xN data with M2 observations of N variables
% OUTPUTS:
%         qvec - Nx1 unit direction of maximum separation

    qvec = ones(size(X1,2), 1);
    xbar1 = mean(X1);
    xbar2 = mean(X2);

    % Compute the within-class means and scatter matrices
    % %s
    X_bar = xbar1 + xbar2;
    % calculating the zero mean matrices for X1 and X2
    X1_zeromean = X1 - ones(length(X1), 1) * mean(X1, 1); 
    X2_zeromean = X2 - ones(length(X2), 1) * mean(X2, 1);
    
    % calculating the within label scatter matrices associated with the partitioning X1
    % and X2
    S_1 = X1_zeromean' * X1_zeromean;
    S_2 = X2_zeromean' * X2_zeromean;
    S_w = S_1 + S_2;
    
    % Compute the between-class scatter matrix
    Sb_part = [xbar1 - X_bar; xbar2 - X_bar];
    S_b = Sb_part' * Sb_part;
    
    % Fisher's linear discriminant is the largest eigenvector
    % of the Rayleigh quotient
    % %
    % % COMPUTE qvec
    % %
    FisherDA = S_w\S_b;
    [qvec, ~] = eigs(FisherDA, 1);

    % May need to correct the sign of qvec to point towards mean of X1
    if (xbar1 - xbar2)*qvec < 0
        qvec = -qvec;
    end
% END OF FUNCTION
end

function [fpr tpr auc bopt] = roccurve(yvec_in,zvec_in)
% [TPR FPR AUC BOPT]=ROCCURVE(YVEC,ZVEC) computes the
% ROC curve and related values for labels YVEC and scores ZVEC.
% Unique scores are used as thresholds for binary classification.
%
% INPUTS:
%         YVEC - Mx1 labels, +/- computed as ==/~= 1
%         ZVEC - Mx1 scores, real numbers
% OUTPUTS:
%         TPR  - Kx1 vector of  True Positive Rate values
%         FPR  - Kx1 vector of False Positive Rate values
%         AUC  - scalar, Area Under Curve of ROC determined by TPR and FPR
%         BOPT - scalar, optimal threshold for accuracy

    % Sort the scores and permute the labels accordingly
    [zvec, zndx] = sort(zvec_in);
    yvec = yvec_in(zndx);
        
    % Sort and find a unique subset of the scores; problem size
    bvec = unique(zvec);
    bm = numel(bvec);
    
    % Compute a confusion matrix for each unique threshold value;
    % extract normalized entries into TPR and FPR vectors; track
    % the accuracy and optimal B threshold
    tpr = [];
    fpr = [];
    acc = -inf;
    bopt = -inf;
%     for jx = 1:bm
        % %
        % % FIND TPR, FPR, OPTIMAL THRESHOLD
        % %
        
        % getting the confusion matrix and store the four components in
        % variables to access
    cmat = confmat(yvec, zvec,0);
    tp = cmat(1,1);
    fn = cmat(1,2);
    fp = cmat(2,1);
    tn = cmat(2,2);

    % calculating positive instances and negative instances 
    p = tp + fn;
    n = fp + tn;

    % appending new TPR elements and FPR elements into the array
    tpr = [tpr tp/p];
    fpr = [fpr fp/n];
    % threshold that creates higher accuracy will replace the current
    % threshold value 
    newacc = (tp+tn)/(p+n);
    if newacc > acc
        acc = newacc;
%         bopt = bvec(jx);
    end
    
    acc_is = acc
    % Ensure that the rates, from these scores, will plot correctly
    tpr = sort(tpr);
    fpr = sort(fpr);
    
    % Compute AUC for this ROC
    auc = aucofroc(tpr, fpr);
end
    
function cmat = confmat(yvec, zvec, theta)
% CMAT=CONFMAT(YVEC,ZVEC,THETA) finds the confusion matrix CMAT for labels
% YVEC from scores ZVEC and a threshold THETA. YVEC is assumed to be +1/-1
% and each entry of ZVEC is scored as -1 if <THETA and +1 otherwise. CMAT
% is returned as [TP FN ; FP TN]
%
% INPUTS:
%         YVEC  - Mx1 values, +/- computed as ==/~= 1
%         ZVEC  - Mx1 scores, real numbers
%         THETA - threshold real-valued scalar
% OUTPUTS:
%         CMAT  - 2x2 confusion matrix; rows are +/- labels,
%                 columns are +/- classifications

    % Find the plus/minus 1 vector of quantizations
    qvec = sign((zvec >= theta) - 0.5);
    
    % Compute the confusion matrix by entries
    % %
    % % COMPUTE MATRIX
    % %
    
    %initialize a empty confusion matrix with all components set to zero
    cmat = zeros(2,2);
    
    for idx = 1 : size(yvec)
        % comparing the yvec at idx to qvec respectively to adjust
        % confusion matrix's components
        if yvec(idx) == 1 && qvec(idx) == 1
            cmat(1,1) = cmat(1,1) + 1;
        elseif yvec(idx) == 1 && qvec(idx) == -1
            cmat(1,2) = cmat(1,2) + 1;
        elseif yvec(idx) == -1 && qvec(idx) == 1
            cmat(2,1) = cmat(2,1) + 1;
        else
            cmat(2,2) = cmat(2,2) + 1;
        end
    end
end

function auc = aucofroc(tpr, fpr)
% AUC=AUCOFROC(TPR,FPR) finds the Area Under Curve of the
% ROC curve specified by the TPR, True Positive Rate, and
% the FPR, False Positive Rate.
%
% INPUTS:
%         TPR - Kx1 vector, rate for underlying score threshold 
%         FPR - Kx1 vector, rate for underlying score threshold 
% OUTPUTS:
%         AUC - integral, from Trapezoidal Rule on [0,0] to [1,1]

    [X undx] = sort(reshape(tpr, 1, numel(tpr)));
    Y = sort(reshape(fpr(undx), 1, numel(undx)));
    auc = abs(trapz([0 Y 1] , [0 X 1]));
end