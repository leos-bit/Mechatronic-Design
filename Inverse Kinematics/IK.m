function [theta1, theta2, theta3, valid] = IK(EndPosn)
    % inverse kinematics maps end position to motor angles
    % GIVES THETA IN DEGREES
    x = EndPosn(1); y = EndPosn(2); z = EndPosn(3);

    % get geometry
    [l,L,sb,sp,wb,wp,ub,up] = geom;
    
    % define some terms for alegbra
    a = wb-up;
    b = (sp/2) - (sqrt(3)/2)*wb;
    c = wp - (1/2)*wb;

    E1 = 2*L*(y+a);
    F1 = 2*z*L;
    G1 = x^2 + y^2 + z^2 + a^2 + L^2 + 2*y*a - l^2;

    E2 = -L*((sqrt(3)*(x+b))+y+c);
    F2 = 2*z*L;
    G2 = x^2 + y^2 + z^2 + b^2 + c^2 + L^2 + 2*(x*b+y*c) - l^2;

    E3 = L*((sqrt(3)*(x-b))-y-c);
    F3 = 2*z*L;
    G3 = x^2 + y^2 + z^2 + b^2 + c^2 + L^2 + 2*(-x*b+y*c) - l^2;

    t1_plus = (-F1+sqrt(E1^2 + F1^2 - G1^2))/(G1-E1);
    t1_minus = (-F1-sqrt(E1^2 + F1^2 - G1^2))/(G1-E1);

    t2_plus = (-F2+sqrt(E2^2 + F2^2 - G2^2))/(G2-E2);
    t2_minus = (-F2-sqrt(E2^2 + F2^2 - G2^2))/(G2-E2);

    t3_plus = (-F3+sqrt(E3^2 + F3^2 - G3^2))/(G3-E3);
    t3_minus = (-F3-sqrt(E3^2 + F3^2 - G3^2))/(G3-E3);

    theta1_plus = 2*atand(t1_plus);
    theta1_minus = 2*atand(t1_minus);

    theta2_plus = 2*atand(t2_plus);
    theta2_minus = 2*atand(t2_minus);
    
    theta3_plus = 2*atand(t3_plus);
    theta3_minus = 2*atand(t3_minus);

    %disp([theta1_plus,theta2_plus,theta3_plus]);
    %disp([theta1_minus,theta2_minus,theta3_minus]);

    [solnVec,solnNum,valid] = decider(theta1_plus,theta1_minus,theta2_plus,theta2_minus,theta3_plus,theta3_minus);

   % disp("solnNum:"+solnNum)

    %solnVec
    
   % disp("valid "+valid)

    if valid == true
        theta1 = solnVec(1,:,1);
        theta2 = solnVec(2,:,1);
        theta3 = solnVec(3,:,1);
    else
        theta1 = [];
        theta2 = [];
        theta3 = [];
    end
end