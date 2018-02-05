import { BrowserModule } from '@angular/platform-browser';
import { NgModule } from '@angular/core';


import { AppComponent } from './app.component';
import { AppRoutingModule } from './/app-routing.module';
import { BrowserAnimationsModule } from '@angular/platform-browser/animations';
import { NavigationComponent } from './navigation/navigation.component';
import { FullyConnectedNnComponent } from './fully-connected-nn/fully-connected-nn.component';
import { MainComponent } from './main/main.component';
import { HyperparametersComponent } from './fully-connected-nn/hyperparameters/hyperparameters.component';
import { NetworkVisualizationComponent } from './fully-connected-nn/network-visualization/network-visualization.component';
import { TrainingVisualizationComponent } from './fully-connected-nn/training-visualization/training-visualization.component';
import { MaterialDesignModule } from './material-design/material-design.module';
import { LayerComponent } from './fully-connected-nn/hyperparameters/layer/layer.component';


@NgModule({
  declarations: [
    AppComponent,
    NavigationComponent,
    FullyConnectedNnComponent,
    MainComponent,
    HyperparametersComponent,
    NetworkVisualizationComponent,
    TrainingVisualizationComponent,
    LayerComponent
  ],
  imports: [
    BrowserModule,
    AppRoutingModule,
    BrowserAnimationsModule,
    MaterialDesignModule
  ],
  providers: [],
  bootstrap: [AppComponent]
})
export class AppModule { }
