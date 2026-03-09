function [topTriangle,bottomTriangle,Arm1,Arm2,Arm3] = GetDrawingVectors(theta1,theta2,theta3,EndPosn)
    % this function gives us the vectors for the 3D plot
    % TAKES THETA IN DEGREES

    % get geometry from geom
    [l,L,sb,sp,wb,wp,ub,up] = geom;

    % get coordinates for end effector:
    x = EndPosn(1); y = EndPosn(2); z = EndPosn(3);

    % fixed top triangle vertex coordinates
    topTriangle.XData = [sb/2, 0, -sb/2, sb/2];
    topTriangle.YData = [-wb, ub, -wb, -wb];
    topTriangle.ZData = [0, 0, 0, 0];
                       
    % end effector vertex coordinates:
    bottomTriangle.XData = [0+x, sp/2+x, -sp/2+x, 0+x];
    bottomTriangle.YData = [-up+y, wp+y, wp+y, -up+y];
    bottomTriangle.ZData = [z, z, z, z];

    % arm coordinates:
    Arm1.XData = [0, 0, x];
    Arm1.YData = [-wb, -wb-L*cosd(theta1), -up+y];
    Arm1.ZData = [0, -L*sind(theta1), z];

    Arm2.XData = [sqrt(3)/2*wb,(sqrt(3)/2)*(wb+L*cosd(theta2)), sp/2+x];
    Arm2.YData = [(1/2)*wb,(1/2)*(wb+L*cosd(theta2)), wp+y];
    Arm2.ZData = [0,-L*sind(theta2),z];
    
    Arm3.XData = [(-sqrt(3)/2)*wb,-(sqrt(3)/2)*(wb+L*cosd(theta3)),-sp/2+x];
    Arm3.YData = [(1/2)*wb,(1/2)*(wb+L*cosd(theta3)),wp+y];
    Arm3.ZData = [0,-L*sind(theta3),z];
end