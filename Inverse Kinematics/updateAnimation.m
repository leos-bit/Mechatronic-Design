function updateAnimation(theta1,theta2,theta3,EndPosn)
        [~,bottomTriangle,Arm1,Arm2,Arm3] = GetDrawingVectors(theta1,theta2,theta3,EndPosn);

        % update plots 
        Arm1_plot.XData = Arm1.XData;
        Arm1_plot.YData = Arm1.YData;
        Arm1_plot.ZData = Arm1.ZData;

        Arm2_plot.XData = Arm2.XData;
        Arm2_plot.YData = Arm2.YData;
        Arm2_plot.ZData = Arm2.ZData;

        Arm3_plot.XData = Arm3.XData;
        Arm3_plot.YData = Arm3.YData;
        Arm3_plot.ZData = Arm3.ZData;

        bottom_triangle_plot.XData = bottomTriangle.XData;
        bottom_triangle_plot.YData = bottomTriangle.YData;
        bottom_triangle_plot.ZData = bottomTriangle.ZData;

        drawnow
end