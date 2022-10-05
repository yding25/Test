%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%% Here, we apply AI planning techniques to atuonomous driving.    %%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

step(0..n).

1{ 
    visit_grocery(I);
    visit_gas(I);
    visit_school(I);
    visit_home(I)
}1 :- step(I), I>=0, I<n.

% cannot visit one location at different steps
:-visit_grocery(I1), visit_grocery(I2), I1!=I2, step(I1), step(I2), I1>=0, I1<n, I2>=0, I2<n.
:-visit_gas(I1), visit_gas(I2), I1!=I2, step(I1), step(I2), I1>=0, I1<n, I2>=0, I2<n.
:-visit_school(I1), visit_school(I2), I1!=I2, step(I1), step(I2), I1>=0, I1<n, I2>=0, I2<n.
:-visit_home(I1), visit_home(I2), I1!=I2, step(I1), step(I2), I1>=0, I1<n, I2>=0, I2<n.

1{ 
    grocery1(I);
    grocery2(I)
}1 :- visit_grocery(I), step(I), I>=0, I<n.

1{ 
    gas1(I);
    gas2(I)
}1 :- visit_gas(I), step(I), I>=0, I<n.

#show visit_grocery/1.
#show grocery1/1.
#show grocery2/1.

#show visit_gas/1.
#show gas1/1.
#show gas2/1.

#show visit_school/1.
#show visit_home/1.