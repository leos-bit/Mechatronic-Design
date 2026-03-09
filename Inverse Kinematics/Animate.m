function [] = Animate(theta1_vec,theta2_vec,theta3_vec)
    % THETA IN DEGREES

    for index = 1:size(theta1_vec,2)
        % get end position for each set of thetas with forward kinematics:
        EndPosn(:,:,index) = FK(theta1_vec(index),theta2_vec(index),theta3_vec(index));

        % get drawing vectors for plot for each set of thetas 
        [topTriangle(index),bottomTriangle(index),Arm1(index),Arm2(index),Arm3(index)] = GetDrawingVectors(theta1_vec(index),theta2_vec(index),theta3_vec(index),EndPosn(:,:,index));
    end

    % initialize plot
    %clc
    %figure(2), clf, 
    hold on
    axis equal
    axis ([-500 500 -500 500 -1000 300])
    view(-290, 25)
    grid on, box on
    %title('\bfForward Kinematics')
    xlabel('x (mm)'), ylabel('y (mm)'), zlabel('z (mm)')

    % Draw first position:
    top_triangle_plot = plot3([topTriangle(1).XData],[topTriangle(1).YData],[topTriangle(1).ZData],'.k-', 'LineWidth', 1, 'MarkerSize', 1);
    bottom_triangle_plot = plot3([bottomTriangle(1).XData],[bottomTriangle(1).YData],[bottomTriangle(1).ZData],'.k-', 'LineWidth', 1, 'MarkerSize', 1);

    Arm1_plot = plot3([Arm1(1).XData],[Arm1(1).YData], [Arm1(1).ZData],'.k-', 'LineWidth', 1, 'MarkerSize', 10);
    Arm2_plot = plot3([Arm2(1).XData],[Arm2(1).YData], [Arm2(1).ZData],'.k-', 'LineWidth', 1, 'MarkerSize', 10);
    Arm3_plot = plot3([Arm3(1).XData],[Arm3(1).YData], [Arm3(1).ZData],'.k-', 'LineWidth', 1, 'MarkerSize', 10);

    % now animate for all posn in theta vectors
    for tx = 1:size(theta1_vec,2)
        pause(0.03)
        % update plots 
        Arm1_plot.XData = Arm1(tx).XData;
        Arm1_plot.YData = Arm1(tx).YData;
        Arm1_plot.ZData = Arm1(tx).ZData;

        Arm2_plot.XData = Arm2(tx).XData;
        Arm2_plot.YData = Arm2(tx).YData;
        Arm2_plot.ZData = Arm2(tx).ZData;

        Arm3_plot.XData = Arm3(tx).XData;
        Arm3_plot.YData = Arm3(tx).YData;
        Arm3_plot.ZData = Arm3(tx).ZData;

        bottom_triangle_plot.XData = bottomTriangle(tx).XData;
        bottom_triangle_plot.YData = bottomTriangle(tx).YData;
        bottom_triangle_plot.ZData = bottomTriangle(tx).ZData;

        drawnow
    end
    hold off
end