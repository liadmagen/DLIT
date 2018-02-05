import { NgModule } from '@angular/core';
// import { CommonModule } from '@angular/common';

import {MatButtonModule, MatCheckboxModule, MatSliderModule} from '@angular/material';


@NgModule({
  imports: [
    // CommonModule,
    MatButtonModule,
    MatCheckboxModule,
    MatSliderModule
  ],
  exports: [
    MatButtonModule,
    MatCheckboxModule,
    MatSliderModule
  ]
})
export class MaterialDesignModule { }
