function [] = plotSkeleton2D(Pos, color, lineWidth, invertY, scaleFactor, plotOnlyTrackedJoints)
% plot skeleton of one person from the BIWI RGBD-ID dataset
%
% Author: Matteo Munaro
% Date: 27th July 2013

hold on;

% Plot points:
x = scaleFactor*Pos(:,5);
y = scaleFactor*Pos(:,6);
if invertY
    y = scaleFactor*480 - y;
end

% Plot lines:
if length(color) == 3
    if plotOnlyTrackedJoints
        plot(x(Pos(:,7)==2),y(Pos(:,7)==2),'y.','MarkerSize',30);   % plot only tracked joints
    else
        plot(x,y,'y.','MarkerSize',30);
    end

    plot(x([4 3]),y([4 3]),'Color',color,'LineWidth',lineWidth);
    plot(x([3 5 6 7 8 ]),y([3 5 6 7 8 ]),'Color',color,'LineWidth',lineWidth);
    plot(x([3 9 10 11 12]),y([3 9 10 11 12]),'Color',color,'LineWidth',lineWidth);
    plot(x([3 2 1]),y([3 2 1]),'Color',color,'LineWidth',lineWidth);
    plot(x([1 13 14 15 16]),y([1 13 14 15 16]),'Color',color,'LineWidth',lineWidth);
    plot(x([1 17 18 19 20]),y([1 17 18 19 20]),'Color',color,'LineWidth',lineWidth);
else
    if plotOnlyTrackedJoints
        plot(x(Pos(:,7)==2),y(Pos(:,7)==2),'y.','MarkerSize',30);   % plot only tracked joints
    else
        plot(x,y,'y.','MarkerSize',30);
    end

    plot(x([4 3]),y([4 3]),'g','LineWidth',lineWidth);
    plot(x([3 5 6 7 8 ]),y([3 5 6 7 8 ]),'c','LineWidth',lineWidth);
    plot(x([3 9 10 11 12]),y([3 9 10 11 12]),'m','LineWidth',lineWidth);
    plot(x([3 2 1]),y([3 2 1]),'k','LineWidth',lineWidth);
    plot(x([1 13 14 15 16]),y([1 13 14 15 16]),'r','LineWidth',lineWidth);
    plot(x([1 17 18 19 20]),y([1 17 18 19 20]),'r','LineWidth',lineWidth);
end
