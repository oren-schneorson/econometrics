function [ dt, comp_trend_seasonal, comp_Irr ] = sa_adj( y, s )
%SA_ADJ Seasonsally adjusts any vector of data
%   Detailed explanation goes here

% based on:
% https://www.mathworks.com/help/econ/seasonal-adjustment-using-snxd7m-seasonal-filters.html




% Symmetric weights
sW3 = [1/9;2/9;1/3;2/9;1/9];

sW5 = [1/15;2/15;repmat(1/5,3,1);2/15;1/15];   

sW13 = [1/24;repmat(1/12,11,1);1/24];

sW5H = [-0.073, 0.294, 0.558, 0.294, -0.073]';

sW7H = [-0.059, 0.059, 0.294, 0.412, 0.294, 0.059, -0.059]';

sW9H = [-0.041, -0.010, 0.119, 0.267, 0.330, 0.267, 0.119, -0.010, -0.041]';


sW13H = [-0.019, -0.028, 0, .066, .147, .214,...
      .24, .214, .147, .066, 0, -0.028, -0.019]';
  
sW23H = [-0.004, -0.011, -0.016, -0.015, -0.005, 0.013, 0.039, 0.068, 0.097,...
    0.122, 0.138, 0.148, 0.138, 0.122, 0.097, 0.068, 0.039, 0.013, -0.005,...
    -0.015, -0.016, -0.011, -0.004]';

% Asymmetric weights for end of series
aW3 = [.259 .407;.37 .407;.259 .185;.111 0];
aW3H = [.259 .407;.37 .407;.259 .185;.111 0]';


aW5 = [.150 .250 .293;
       .217 .250 .283;
       .217 .250 .283;
       .217 .183 .150;
       .133 .067    0;
       .067   0     0];

aW5H = [  0      -.073   .403    .670;
        -.073    .294   .522    .257]';

% aW7 = ?
aW9H = [ 0       0       0      -.156   -.034   .185    .424    .581;
        0       0       -.049  -.011    .126   .282    .354    .298;
        0       -.022   0       .120    .259   .315    .242    .086;
        -.031   -.004   .120    .263    .324   .255    .102   -.029]';

aW13H = [-.034  -.017   .045   .148   .279   .421;
       -.005   .051   .130   .215   .292   .353;
        .061   .135   .201   .241   .254   .244;
        .144   .205   .230   .216   .174   .120;
        .211   .233   .208   .149   .080   .012;
        .238   .210   .144   .068   .002  -.058;
        .213   .146   .066   .003  -.039  -.092;
        .147   .066   .004  -.025  -.042  0    ;
        .066   .003  -.020  -.016  0      0    ;
        .001  -.022  -.008  0      0      0    ;
       -.026  -.011   0     0      0      0    ;
       -.016   0      0     0      0      0    ];

aW23H = [...
    0     0     0     0     0     0     0     0     0     0     -.077;
    0     0     0     0     0     0     0     0     0     -.046 -.041;
    0     0     0     0     0     0     0     0     -.022 -.025 -.025;
    0     0     0     0     0     0     0     -.008 -.014 -.018 -.015;
    0     0     0     0     0     0     -.001 -.008 -.013 -.012 -.003;
    0     0     0     0     0      .003 -.006 -.011 -.011 -.002  .015;
    0     0     0     0      .002 -.006 -.012 -.011 -.003  .015  .039;
    0     0     0      .001 -.007 -.013 -.011 -.003  .015  .039  .068;
    0     0     -.002 -.007 -.013 -.013 -.003  .014  .039  .068  .097;
    0     -.003 -.010 -.015 -.014 -.005  .014  .040  .069  .097  .122;
    -.004 -.011 -.016 -.015 -.005  .013  .039  .068  .097  .122  .138]';


% Henderson filter, number of terms
if s == 4
    nterms = 5;
elseif s == 12
    nterms = 13;
end

T = size(y, 1);

% Detrend the data using a s+1-term moving average
if s == 4
    sW = [1/8; repmat(1/4,3,1); 1/8];
elseif s == 12
    sW = sW13;
end

yS = conv(y, sW, 'same');
yS(1:s/2) = yS(s/2+1);
yS(T-s/2+1:T) = yS(T-s/2);
xt = y./yS;


% Create seasonal indices
sidx = cell(s,1); % Preallocation
for i = 1:s
 sidx{i,1} = i:s:T;
end


% S3x3 seasonal filter
% Apply filter to each period
shat = NaN*y;
for i = 1:s
    ns = length(sidx{i});
    first = 1:4;
    last = ns - 3:ns;
    dat = xt(sidx{i});
    
    sd = conv(dat, sW3, 'same');
    sd(1:2) = conv2(dat(first), 1, rot90(aW3,2),'valid');
    
    sd(ns-1:ns) = conv2(dat(last),1, aW3,'valid');
    shat(sidx{i}) = sd;
    
end


if s == 4
    % 5-term moving average of filtered series
    sW = [1/8; repmat(1/4, 3, 1); 1/8];
elseif s==12
    % 13-term moving average of filtered series
    sW = sW13;   
end

sb = conv(shat, sW, 'same');
sb(1:s/2) = sb(s+1:s+s/2); 
sb(T-s/2+1:T) = sb(T-s-s/2+1:T-s);

% Center to get final estimate
s33 = shat./sb;

% Deseasonalize series
dt = y./s33;



% Henderson filter weights
% sWH: Symmetric weights for end of series
% aWH: Asymmetric weights for end of series


if nterms == 23
    
sWH = sW23H;
aWH = aW23H;    

elseif nterms == 13

sWH = sW13H;    
aWH = aW13H;

elseif nterms == 9

sWH = sW9H;
awH = aW9H;

elseif nterms == 7

sWH = sW7H;
% aWH = ?;

elseif nterms == 5

sWH = sW5H;
aWH = aW5H;

end



% Apply H-term Henderson filter
first = 1:s;
last = T-s+1:T;
hs = conv(dt, sWH, 'same'); % Henderson filtered data
hs(T-s/2+1:end) = conv2(dt(last), 1, aWH, 'valid');
hs(1:s/2) = conv2(dt(first), 1, rot90(aWH, 2), 'valid');
   
% New detrended series
xt = y./hs;




% S3x5 seasonal filter 
% Apply filter to each period
shat = NaN*y;
for i = 1:s
    ns = length(sidx{i});
    first = 1:6;
    last = ns-5:ns;
    dat = xt(sidx{i});
    
    sd = conv(dat, sW5, 'same');
    sd(1:3) = conv2(dat(first), 1, rot90(aW5,2), 'valid');
    sd(ns-2:ns) = conv2(dat(last), 1, aW5, 'valid');
    shat(sidx{i}) = sd;
end


% H-term moving average of filtered series
if s == 12
    sW = sW13;
elseif s == 4
    sW = sW5;
end
% H-term moving average of filtered series
sb = conv(shat, sW,'same');
sb(1:s/2) = sb(s+1:s+s/2);
sb(T-s/2+1:T) = sb(T-s-s/2+1:T-s);


% Center to get final estimate
s35 = shat./sb;
% Deseasonalized series
dt = y./s35;


comp_trend_seasonal = hs.*s35;
comp_Irr = dt./hs;



end

