classdef animation < handle

    properties
        top_triangle_plot 
        bottom_triangle_plot 
        Arm1_plot 
        Arm2_plot 
        Arm3_plot
        
        topTriangle
        bottomTriangle
        Arm1
        Arm2
        Arm3

        this1
        this2
    end
    methods

        function obj = animation(theta1,theta2,theta3)
            
                % get end position for each set of thetas with forward kinematics:
            
                [EndPosn,~] = FK(theta1,theta2,theta3);

                % get drawing vectors for plot for each set of thetas 
                [obj.topTriangle,obj.bottomTriangle,obj.Arm1,obj.Arm2,obj.Arm3] = GetDrawingVectors(theta1,theta2,theta3,EndPosn);
            

       end
       function obj = Animate(obj,theta1,theta2,theta3)
            % THETA IN DEGREES

            %obj = animation(theta1,theta2,theta3)
            
            % initialize plot
            clc
            %hold on
            %figure(1)

            fig = figure(1);
            
            hold on
            axis equal
           
            %cla(this)
            axis equal
            axis ([-500 500 -500 500 -1000 300])
            view(290, 35)
            grid on, box on
            %title('\bfForward Kinematics')
            xlabel('x (mm)'), ylabel('y (mm)'), zlabel('z (mm)')

            % Draw first position:
            obj.top_triangle_plot = plot3([obj.topTriangle.XData],[obj.topTriangle.YData],[obj.topTriangle.ZData],'.k-', 'LineWidth', 1, 'MarkerSize', 1);
            obj.bottom_triangle_plot = plot3([obj.bottomTriangle.XData],[obj.bottomTriangle.YData],[obj.bottomTriangle.ZData],'.k-', 'LineWidth', 1, 'MarkerSize', 1);

            obj.Arm1_plot = plot3([obj.Arm1.XData],[obj.Arm1.YData], [obj.Arm1.ZData],'.k-', 'LineWidth', 1, 'MarkerSize', 10);
            obj.Arm2_plot = plot3([obj.Arm2.XData],[obj.Arm2.YData], [obj.Arm2.ZData],'.k-', 'LineWidth', 1, 'MarkerSize', 10);
            obj.Arm3_plot = plot3([obj.Arm3.XData],[obj.Arm3.YData], [obj.Arm3.ZData],'.k-', 'LineWidth', 1, 'MarkerSize', 10);

            %drawnow
    end

    function obj = updateAnimation(obj,theta1,theta2,theta3,EndPosn)
        [~,obj.bottomTriangle,obj.Arm1,obj.Arm2,obj.Arm3] = GetDrawingVectors(theta1,theta2,theta3,EndPosn);

        % update plots
        obj.Arm1_plot.XData = obj.Arm1.XData;
        obj.Arm1_plot.YData = obj.Arm1.YData;
        obj.Arm1_plot.ZData = obj.Arm1.ZData;

        obj.Arm2_plot.XData = obj.Arm2.XData;
        obj.Arm2_plot.YData = obj.Arm2.YData;
        obj.Arm2_plot.ZData = obj.Arm2.ZData;

        obj.Arm3_plot.XData = obj.Arm3.XData;
        obj.Arm3_plot.YData = obj.Arm3.YData;
        obj.Arm3_plot.ZData = obj.Arm3.ZData;

        obj.bottom_triangle_plot.XData = obj.bottomTriangle.XData;
        obj.bottom_triangle_plot.YData = obj.bottomTriangle.YData;
        obj.bottom_triangle_plot.ZData = obj.bottomTriangle.ZData;

        drawnow
    end
    end

end
