function [solnVec,solnNum,valid] = decider(theta1_plus,theta1_minus,theta2_plus,theta2_minus,theta3_plus,theta3_minus)

    valid = false;

    solnNum = 0;
    solnVec = [];

    [Soln1,valid1] = FK(theta1_plus,theta2_plus,theta3_plus);
    if valid1 == true && all([theta1_plus,theta2_plus,theta3_plus]<90) && all([theta1_plus,theta2_plus,theta3_plus]>-90)
        solnNum = solnNum+1;
        solnVec(:,:,solnNum) = [theta1_plus,theta2_plus,theta3_plus]';
    end

    [Soln2,valid2] = FK(theta1_plus,theta2_plus,theta3_minus);
    if valid2 == true && all([theta1_plus,theta2_plus,theta3_minus]<90) && all([theta1_plus,theta2_plus,theta3_minus]>-90)
        solnNum = solnNum+1;
        solnVec(:,:,solnNum) = [theta1_plus,theta2_plus,theta3_minus]';
    end

    [Soln3,valid3] = FK(theta1_plus,theta2_minus,theta3_plus);
    if valid3 == true && all([theta1_plus,theta2_minus,theta3_plus]<90) && all([theta1_plus,theta2_minus,theta3_plus]>-90)
        solnNum = solnNum+1;
        solnVec(:,:,solnNum) = [theta1_plus,theta2_minus,theta3_plus]';
    end

    [Soln4,valid4] = FK(theta1_plus,theta2_minus,theta3_minus);
    if valid4 == true && all([theta1_plus,theta2_minus,theta3_minus]<90) && all([theta1_plus,theta2_minus,theta3_minus]>-90)
        solnNum = solnNum+1;
        solnVec(:,:,solnNum) = [theta1_plus,theta2_minus,theta3_minus]';
    end

    [Soln5,valid5] = FK(theta1_minus,theta2_plus,theta3_plus);
    if valid5 == true && all([theta1_minus,theta2_plus,theta3_plus]<90)&& all([theta1_minus,theta2_plus,theta3_plus]>-90)
        solnNum = solnNum+1;
        solnVec(:,:,solnNum) = [theta1_minus,theta2_plus,theta3_plus]';
    end

    [Soln6,valid6] = FK(theta1_minus,theta2_plus,theta3_minus);
    if valid6 == true && all([theta1_minus,theta2_plus,theta3_minus]<90)&& all([theta1_minus,theta2_plus,theta3_minus]>-90)
        solnNum = solnNum+1;
        solnVec(:,:,solnNum) = [theta1_minus,theta2_plus,theta3_minus]';
    end

    [Soln7,valid7] = FK(theta1_minus,theta2_minus,theta3_plus);
    if valid7 == true && all([theta1_minus,theta2_minus,theta3_plus]<90)&& all([theta1_minus,theta2_minus,theta3_plus]>-90)
        solnNum = solnNum+1;
        solnVec(:,:,solnNum) = [theta1_minus,theta2_minus,theta3_plus]';
    end

    [Soln8,valid8] = FK(theta1_minus,theta2_minus,theta3_minus);
    if valid8 == true && all([theta1_minus,theta2_minus,theta3_minus]<90) && all([theta1_minus,theta2_minus,theta3_minus]>-90)
        solnNum = solnNum+1;
        solnVec(:,:,solnNum) = [theta1_minus,theta2_minus,theta3_minus]';
    end

    if solnNum > 0
        valid = true;
    end
end