from math import sqrt, atan2, degrees
from itertools import permutations
l = 533.4
L = 304.8
sb = 265.7
sp = 127
wb = (sqrt(3)/6)*sb
ub = (sqrt(3)/3)*sb
wp = (sqrt(3)/6)*sp
up = (sqrt(3)/3)*sp

def decider(theta1_plus,theta1_minus,theta2_plus,theta2_minus,theta3_plus,theta3_minus):
    valid = False

    solnNum = 0
    solnVec = []

    Soln1,valid1 = FK(theta1_plus,theta2_plus,theta3_plus)
    if (valid1 == true and 
        theta1_plus < 90 and theta2_plus < 90 and theta3_plus < 90 and 
        theta1_plus > -90 and theta2_plus > -90 and theta3_plus >-90):
        solnNum = solnNum+1
        solnVec(:,:,solnNum) = [theta1_plus,theta2_plus,theta3_plus]

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

def atand(x):
    return degrees(atan2(x))

def getAngles(arm, x, y, z):
    a = wb-up
    b = (sp/2) - (sqrt(3)/2)*wb
    c = wp - (1/2)*wb
    E1 = 2*L*(y+a)
    F1 = 2*z*L
    G1 = x**2 + y**2 + z**2 + a**2 + L**2 + 2*y*a - l**2

    E2 = -L*((sqrt(3)*(x+b))+y+c)
    F2 = 2*z*L
    G2 = x**2 + y**2 + z**2 + b**2 + c**2 + L**2 + 2*(x*b+y*c) - l**2

    E3 = L*((sqrt(3)*(x-b))-y-c)
    F3 = 2*z*L
    G3 = x**2 + y**2 + z**2 + b**2 + c**2 + L**2 + 2*(-x*b+y*c) - l**2

    t1_plus = (-F1+sqrt(E1**2 + F1**2 - G1**2))/(G1-E1)
    t1_minus = (-F1-sqrt(E1**2 + F1**2 - G1**2))/(G1-E1)

    t2_plus = (-F2+sqrt(E2**2 + F2**2 - G2**2))/(G2-E2)
    t2_minus = (-F2-sqrt(E2**2 + F2**2 - G2**2))/(G2-E2)

    t3_plus = (-F3+sqrt(E3**2 + F3**2 - G3**2))/(G3-E3)
    t3_minus = (-F3-sqrt(E3**2 + F3**2 - G3**2))/(G3-E3)

    theta1_plus = 2*atand(t1_plus)
    theta1_minus = 2*atand(t1_minus)

    theta2_plus = 2*atand(t2_plus)
    theta2_minus = 2*atand(t2_minus)
    
    theta3_plus = 2*atand(t3_plus)
    theta3_minus = 2*atand(t3_minus)

    solnVec,solnNum,valid = decider(theta1_plus,theta1_minus,theta2_plus,theta2_minus,theta3_plus,theta3_minus)

    if valid == true
        theta1 = solnVec(1,:,1)
        theta2 = solnVec(2,:,1)
        theta3 = solnVec(3,:,1)
    else
        theta1 = []
        theta2 = []
        theta3 = []
    end

def main():
    arm = ArmSpecs()

if __name__ == "__main__":
    main()
