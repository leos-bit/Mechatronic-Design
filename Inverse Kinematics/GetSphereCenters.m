function [A1v, A2v, A3v, r1, r2, r3] = GetSphereCenters(theta1,theta2,theta3)
    % TAKES THETA IN DEGREES
    
    % get robot geometry from geom
    [l,L,~,sp,wb,wp,~,up] = geom;

    r1 = l; r2 = l; r3 = l; % since legs all same length, etc

    % write absolute vectors to centers of spheres
    A1v = [0, -wb-L*cosd(theta1)+up, -L*sind(theta1)]';
    A2v = [(sqrt(3)/2)*(wb + L*cosd(theta2))-sp/2, (1/2)*(wb + L*cosd(theta2))-wp, -L*sind(theta2)];
    A3v = [-(sqrt(3)/2)*(wb + L*cosd(theta3))+sp/2, (1/2)*(wb + L*cosd(theta3))-wp, -L*sind(theta3)];
end