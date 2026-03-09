function [PlusSoln,MinusSoln,valid] = ThreeSpheres(A1v, A2v, A3v, r1, r2, r3)
    allSame = false;
    valid = true;
    % deal with singularites when 1,3 or 2,3 have same z
    if A3v(3) == A2v(3)
        if A3v(3) == A1v(3)
            allSame = true; % check if we need to go to simplified algo.
        end
        holder = A3v;
        A3v = A1v;
        A1v = holder;
    end
    if A3v(3) == A1v(3)
        holder = A3v;
        A3v = A2v;
        A2v = holder;
    end

    % takes in abs vectors to center of spheres + radius (l = bottom link)
    x1 = A1v(1); y1 = A1v(2); z1 = A1v(3);
    x2 = A2v(1); y2 = A2v(2); z2 = A2v(3);
    x3 = A3v(1); y3 = A3v(2); z3 = A3v(3);

    if allSame == false % if we're not in the special case:
        % doing some alegbra with sphere equations...
        a11 = 2*(x3 - x1);
        a12 = 2*(y3 - y1);
        a13 = 2*(z3 - z1);
        a21 = 2*(x3 - x2);
        a22 = 2*(y3 - y2);
        a23 = 2*(z3 - z2);

        b1 = r1^2 - r3^2 -x1^2 - y1^2 - z1^2 + x3^2 + y3^2 + z3^2;
        b2 = r2^2 - r3^2 -x2^2 - y2^2 - z2^2 + x3^2 + y3^2 + z3^2;

        % doing more algebra ....
        a1 = (a11/a13) - (a21/a23);
        a2 = (a12/a13) - (a22/a23);
        a3 = (b2/a23) - (b1/a13);
    
        a4 = -a2/a1;
        a5 = -a3/a1;

        % more algebra ....
        a6 = (-a21*a4 - a22)/a23;
        a7 = (b2 - a21*a5)/a23;

        % last piece of algebra .....
        a = a4^2 + 1 + a6^2;
        b = 2*a4*(a5 - x1) - 2*y1 + 2*a6*(a7 - z1);
        c = a5*(a5 - 2*x1) + a7*(a7 - 2*z1) + x1^2 + y1^2 - r1^2;

        % get two solns for y:
        yPlusSoln = (-b+sqrt(b^2 - 4*a*c))/(2*a);
        yMinusSoln = (-b-sqrt(b^2 - 4*a*c))/(2*a);

        if b^2 - 4*a*c < 0
            %disp("IMAGINARY SOLUTION")
            valid = false;
        end

        % get two solns for x and z:
        xPlusSoln = a4*yPlusSoln + a5;
        xMinusSoln = a4*yMinusSoln + a5;

        zPlusSoln = a6*yPlusSoln + a7;
        zMinusSoln = a6*yMinusSoln + a7;

        % write final two possible answers
        PlusSoln = [xPlusSoln, yPlusSoln, zPlusSoln]';
        MinusSoln = [xMinusSoln, yMinusSoln, zMinusSoln]';

    else % SPECIAL CASE: 
        % for case where all three knees at same z
        
        % doing some algebra with the sphere equations:
        a = 2*(x3 - x1);
        b = 2*(y3 - y1);
        c = r1^2 - r3^2 - x1^2 - y1^2 + x3^2 + y3^2;
        d = 2*(x3 - x2);
        e = 2*(y3 - y2);
        f = r2^2 - r3^2 - x2^2 - y2^2 + x3^2 + y3^2;
        
        % get unique x and y solutions:
        xSoln = (c*e - b*f)/(a*e - b*d);
        ySoln = (a*f - c*d)/(a*e - b*d);

        % get plus minus solutions for z:
        B = -2*z1;
        C = z1^2 - r1^2 + (xSoln-x1)^2 + (ySoln-y1)^2;

        zPlusSoln = (-B + sqrt(B^2 - 4*C))/2;
        zMinusSoln = (-B - sqrt(B^2 - 4*C))/2;

        if B^2 - 4*C < 0
            %disp("IMAGINARY SOLUTION")
            valid = false;
        end

         % write final two possible answers
        PlusSoln = [xSoln, ySoln, zPlusSoln]';
        MinusSoln = [xSoln, ySoln, zMinusSoln]';
    end
end