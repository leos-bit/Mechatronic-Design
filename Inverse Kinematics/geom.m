function [l,L,sb,sp,wb,wp,ub,up] = geom
    % just a function to easily get robot geometry

    % values in mm
    l = 533.4;
    L = 304.8;
    sb = 265.7;
    sp = 127;

    wb = (sqrt(3)/6)*sb;
    ub = (sqrt(3)/3)*sb;
    wp = (sqrt(3)/6)*sp;
    up = (sqrt(3)/3)*sp;
end