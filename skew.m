clear;
N = 500;
TRIALS = 100;
SIGETA = 40;
SIGEPS = 40;

proj = @(A) A*inv(A.'*A)*A.';

skew = linspace(-0.2,0.2,5);

beta1 = randi(10,10,1) - 5;
beta2 = randi(10,10,1) - 5;
gamma1 = randi(10) - 5;
gamma2 = randi(10,10,1)-5;

results = zeros(length(skew),5);
for k = 1:length(skew)
    gamma_gmm = zeros(TRIALS,1);
    gamma_liml = zeros(TRIALS,1);
    var_gmm = zeros(TRIALS,1);
    var_liml = zeros(TRIALS,1);
    for i = 1:TRIALS
        X1 = randi(100,N,10);
        X2 = randi(100,N,10);
        epsilon = normrnd(0,SIGEPS,N,1);
        epsmod = sinh(asinh(epsilon)-skew(k));
        eta = normrnd(0,SIGETA,N,1);
        Y = X1*beta1 + X2*beta2 + eta;
        y = Y*gamma1 + X1*gamma2 + epsmod;

        % GMM estimation
        Z = [X1 X2];
        Pz = proj(Z);
        X_gmm = [Y X1];
        deltahat = inv(X_gmm.' * Pz * X_gmm) * X_gmm.' * Pz * y;
        gamma_gmm(i) = deltahat(1);
        
        % LIML estimation
        Z = [Y X1];
        M1 = eye(N) - proj(X1);
        Xstar = M1*X2;
        Mxstar = eye(N) - proj(Xstar);
        Mx = M1 * Mxstar;
        Y = [y Y];
        E = eig((Y.' * Mx * Y)^(-1/2) * Y.' * M1 * Y * (Y.' * Mx * Y)^(-1/2));
        kappahat = min(E);
        deltahat = inv(Z.'*(eye(N) - kappahat*Mx)*Z)*Z.'*(eye(N) - kappahat*Mx)*y;
        gamma_liml(i) = deltahat(1);
    end
    
    gmm_bias = sum(mean(gamma_gmm)-gamma1);
    liml_bias = sum(mean(gamma_liml)-gamma1);
    gmm_var = var(gamma_gmm);
    liml_var = var(gamma_liml);
    
    results(k,:) = [skew(k) gmm_bias liml_bias gmm_var liml_var];
end

results
