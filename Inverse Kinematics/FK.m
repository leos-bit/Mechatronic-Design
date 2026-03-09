function [Soln,valid] = FK(theta1,theta2,theta3)
    % this function maps motor angles to end position:
    % TAKES THETA IN DEGREES
  
    % get absolute vectors to sphere centers:
    [A1v, A2v, A3v, r1, r2, r3] = GetSphereCenters(theta1,theta2,theta3);
    
    % get possible solutions from three sphere algorithm:
    [PlusSoln,MinusSoln,valid] = ThreeSpheres(A1v, A2v, A3v, r1, r2, r3);

    if MinusSoln(3) <= 0
        Soln = MinusSoln;
    elseif PlusSoln(3) <= 0
        Soln = PlusSoln;
        %disp("MULTIPLE VALID SOLUTIONS")
        %valid = false;
    end
end