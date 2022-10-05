%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%% Here, we apply AI planning techniques to atuonomous driving.    %%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

step(0..n).
pois(grocery;gas;school;home).

grocery(grocery1).
gas(gas1).
school(schoo1).
home(home1).

ids(grocery1;grocery2;gas1;gas2;schoo1;home1).

1{ 
    visit_grocery(I, ID1);
    visit_gas(I, ID2);
    visit_school(I, ID3);
    visit_home(I, ID4)
}1 :- grocery(ID1), gas(ID2), school(ID3), home(ID4), step(I), I>=0, I<n.

% cannot visit one location at different steps
:-visit_school(I1, ID), visit_school(I2, ID), school(ID), I1!=I2, step(I1), step(I2), I1>=0, I1<n, I2>=0, I2<n.
:-visit_home(I1, ID), visit_home(I2, ID), home(ID), I1!=I2, step(I1), step(I2), I1>=0, I1<n, I2>=0, I2<n.
:-visit_gas(I1, ID), visit_gas(I2, ID), gas(ID), I1!=I2, step(I1), step(I2), I1>=0, I1<n, I2>=0, I2<n.
:-visit_grocery(I1, ID), visit_grocery(I2, ID), grocery(ID), I1!=I2, step(I1), step(I2), I1>=0, I1<n, I2>=0, I2<n.

1{ 
    visit_grocery_info(I, ID1);
}1 :- visit_grocery(I, ID1), grocery(ID1), gas(ID2), school(ID3), home(ID4), step(I), I>=0, I<n.

% cannot visit two location at same steps
%:-visit_gas(I, ID1), visit_gas(I, ID2), ID1!=ID2, gas(ID1), gas(ID2), step(I), I>=0, I<n.

#show visit_grocery/2.
#show visit_gas/2.
#show visit_school/2.
#show visit_home/2.